"""Local Valen execution. No Torch, Transformers, or remote Python code."""

from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np

from .compiler import Compiler
from .errors import ModelCompatibilityError
from .manifest import validate_variant
from .assets import resolve_directory
from .processing import Processor, position_ids
from .media import local_videos
from .protocol import local_images, to_internal, to_public
from .responses import answer
from .schema import candidates


def vision_inputs(pixel_values, grid_thw, config):
    """NumPy adaptation of the validated legacy Qwen vision graph geometry."""
    merge = config["spatial_merge_size"]
    side = math.isqrt(config["num_position_embeddings"])
    indices, weights, coordinates, frame_ids = [], [], [], []
    frame_number = 0
    for frames, height, width in grid_thw.tolist():
        h = np.linspace(0, side - 1, height, dtype=np.float32)
        w = np.linspace(0, side - 1, width, dtype=np.float32)
        hf, wf = h.astype(np.int64), w.astype(np.int64)
        hc, wc = np.minimum(hf + 1, side - 1), np.minimum(wf + 1, side - 1)
        dh, dw = h - hf, w - wf
        corners = [
            hf[:, None] * side + wf[None],
            hf[:, None] * side + wc[None],
            hc[:, None] * side + wf[None],
            hc[:, None] * side + wc[None],
        ]
        values = [
            (1 - dh[:, None]) * (1 - dw[None]),
            (1 - dh[:, None]) * dw[None],
            dh[:, None] * (1 - dw[None]),
            dh[:, None] * dw[None],
        ]

        def reorder(value):
            return (
                np.tile(value, (frames, 1, 1))
                .reshape(frames, height // merge, merge, width // merge, merge)
                .transpose(0, 1, 3, 2, 4)
                .reshape(-1)
            )

        indices.append(np.stack([reorder(v) for v in corners]))
        weights.append(np.stack([reorder(v) for v in values]))
        rows = np.broadcast_to(np.arange(height)[:, None], (height, width))
        cols = np.broadcast_to(np.arange(width)[None], (height, width))
        coordinates.append(np.stack([reorder(rows), reorder(cols)], axis=-1))
        frame_ids.extend(np.repeat(np.arange(frame_number, frame_number + frames), height * width))
        frame_number += frames
    coordinates = np.concatenate(coordinates)
    head_dim = config["hidden_size"] // config["num_heads"]
    inv_freq = 1 / (10000.0 ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2)))
    table = np.outer(np.arange(int(coordinates.max()) + 1, dtype=np.float32), inv_freq)
    rotary = table[coordinates].reshape(len(coordinates), -1)
    rotary = np.concatenate([rotary, rotary], axis=-1)
    frame_ids = np.asarray(frame_ids)
    mask = np.where(frame_ids[:, None] == frame_ids[None], 0, np.finfo(np.float32).min).astype(np.float32)
    return {
        "pixel_values": np.asarray(pixel_values, dtype=np.float32),
        "position_indices": np.concatenate(indices, axis=1).astype(np.int64),
        "position_weights": np.concatenate(weights, axis=1).astype(np.float32),
        "rotary_cos": np.cos(rotary).astype(np.float32),
        "rotary_sin": np.sin(rotary).astype(np.float32),
        "attention_mask": mask,
    }


def features(hidden, branch):
    def pool(span):
        start, end = span
        if not 0 <= start < end <= hidden.shape[1]:
            raise ValueError(f"Invalid role span {span}")
        return hidden[0, start:end].astype(np.float32).mean(axis=0)

    context, instruction = pool(branch.context_span), pool(branch.instruction_span)
    decision = hidden[0, branch.decision_position].astype(np.float32)
    return np.stack(
        [np.stack([context, instruction, pool(span), decision]) for span in branch.candidate_spans]
    )


class ONNXBackend:
    def __init__(
        self,
        model,
        *,
        revision=None,
        local_files_only=False,
        cache_dir=None,
        variant="fp32",
        providers=None,
        num_threads=4,
        execution=None,
        media_kwargs=None,
        max_length=None,
        model_name=None,
        verify=False,
        profile_dir=None,
    ):
        import onnxruntime as ort

        if type(num_threads) is not int or num_threads < 1:
            raise ValueError("num_threads must be positive")
        self.directory = resolve_directory(
            model, revision=revision, local_files_only=local_files_only, cache_dir=cache_dir, variant=variant
        )
        self.config = json.loads((self.directory / "config.json").read_text())
        self.manifest = None
        manifest_path = self.directory / "valen_onnx.json"
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text())
            self.variant = validate_variant(self.manifest, variant, ort.__version__)
            self.graphs = self.variant["graphs"]
            if verify:
                for name, digest in self.variant.get("sha256", {}).items():
                    actual = hashlib.sha256()
                    with (self.directory / name).open("rb") as stream:
                        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            actual.update(chunk)
                    if actual.hexdigest() != digest:
                        raise ModelCompatibilityError(f"Checksum mismatch: {name}")
        else:
            if variant != "fp32":
                raise ModelCompatibilityError("Legacy bundles expose only their original FP32 variant")
            self.variant = {}
            self.graphs = {name: "onnx/" + name + ".onnx" for name in ["text", "head", "vision"]}
        self.name = model_name or (self.manifest or {}).get("model_name", Path(str(model)).name)
        self.execution = execution or self.config.get("qwen_execution", "shared_state")
        self.media_kwargs = deepcopy(self.config.get("valen_media_kwargs", {}))
        if media_kwargs:
            for key, value in media_kwargs.items():
                self.media_kwargs.setdefault(key, {}).update(value)
        if max_length is not None and (type(max_length) is not int or max_length < 1):
            raise ValueError("max_length must be a positive integer")
        self.max_length = max_length if max_length is not None else self.config.get("valen_max_length", 32768)
        self.processor = Processor(self.directory)
        self.providers = providers or ["CPUExecutionProvider"]
        if any((p if isinstance(p, str) else p[0]) == "CUDAExecutionProvider" for p in self.providers):
            ort.preload_dlls()
        available = set(ort.get_available_providers())
        if any(
            (provider if isinstance(provider, str) else provider[0]) not in available
            for provider in self.providers
        ):
            raise ModelCompatibilityError(
                f"Requested providers {self.providers}; installed providers {sorted(available)}"
            )
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = num_threads
        self.options.inter_op_num_threads = 1
        self.options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.options.enable_cpu_mem_arena = False
        if profile_dir is not None:
            Path(profile_dir).mkdir(parents=True, exist_ok=True)
            self.options.enable_profiling = True
            self.options.profile_file_prefix = str(Path(profile_dir) / "valen")
        self.sessions, self.lock, self.closed = {}, threading.RLock(), False

    def session(self, name):
        import onnxruntime as ort

        if self.closed:
            raise RuntimeError("Valen client is closed")
        if name not in self.sessions:
            path = self.directory / self.graphs[name]
            if not path.is_file():
                raise ModelCompatibilityError(f"Missing ONNX graph: {path}")
            try:
                loaded = ort.InferenceSession(str(path), self.options, providers=self.providers)
                requested = [p if isinstance(p, str) else p[0] for p in self.providers]
                if (
                    "CUDAExecutionProvider" in requested
                    and "CUDAExecutionProvider" not in loaded.get_providers()
                ):
                    raise ModelCompatibilityError(
                        "CUDA provider initialization failed; install compatible CUDA/cuDNN runtime dependencies"
                    )
                self.sessions[name] = loaded
            except Exception as exc:
                raise ModelCompatibilityError(
                    f"Cannot load {name} graph with ONNX Runtime {ort.__version__}: {exc}"
                ) from exc
        return self.sessions[name]

    def visual(self, inputs):
        positions, features_ = [], []
        for pixels, grid, token in [
            ("pixel_values", "image_grid_thw", "image_token_id"),
            ("pixel_values_videos", "video_grid_thw", "video_token_id"),
        ]:
            if pixels not in inputs:
                continue
            session = self.session("vision")
            names = {item.name for item in session.get_inputs()}
            feeds = (
                {"pixel_values": inputs[pixels], "image_grid_thw": inputs[grid]}
                if names == {"pixel_values", "image_grid_thw"}
                else vision_inputs(inputs[pixels], inputs[grid], self.config["vision_config"])
            )
            output = session.run(None, feeds)[0]
            where = np.flatnonzero(inputs["input_ids"][0] == self.config[token]).astype(np.int64)
            if len(where) != len(output):
                raise ValueError("Visual placeholder count does not match encoded patches")
            positions.append(where)
            features_.append(output)
        if not positions:
            return np.empty(0, dtype=np.int64), np.empty(
                (0, self.config["text_config"]["hidden_size"]), dtype=np.float32
            )
        return np.concatenate(positions), np.concatenate(features_).astype(np.float32)

    def run(self, inputs, branches):
        ids = np.asarray(inputs["input_ids"], dtype=np.int64)
        if ids.shape[0] != 1 or not np.all(inputs["attention_mask"] == 1):
            raise ValueError("Use one unpadded compiled state at a time")
        positions, visuals = self.visual(inputs)
        rope = position_ids(inputs, self.config["vision_config"]["spatial_merge_size"])
        if "decision" not in self.graphs:
            hidden = self.session("text").run(
                None,
                {
                    "input_ids": ids,
                    "position_ids": rope,
                    "visual_positions": positions,
                    "visual_features": visuals,
                },
            )[0]
            counts = [len(branch.candidate_spans) for branch in branches]
            logits = self.session("head").run(
                None, {"features": np.concatenate([features(hidden, branch) for branch in branches])}
            )[0]
        else:
            embedding_session = self.session("embedding")
            embeds = embedding_session.run(None, {"input_ids": ids})[0]
            embeds[0, positions] = visuals
            session = self.session("decision")
            spans = [
                [
                    branch.context_span,
                    branch.instruction_span,
                    candidate,
                    [branch.decision_position, branch.decision_position + 1],
                ]
                for branch in branches
                for candidate in branch.candidate_spans
            ]
            feeds = {
                "inputs_embeds": embeds,
                "position_ids": rope,
                "attention_mask": np.ones_like(ids),
                "role_spans": np.asarray(spans, dtype=np.int64),
            }
            for item in session.get_inputs():
                if item.name in feeds:
                    continue
                if not item.name.startswith("past"):
                    raise ModelCompatibilityError(f"Unknown graph input {item.name}")
                shape = [
                    dim
                    if isinstance(dim, int)
                    else 1
                    if dim == "batch_size"
                    else 0
                    if dim == "past_sequence_length"
                    else self.config["text_config"]["head_dim"]
                    if dim == "kv_cache_dim"
                    else None
                    for dim in item.shape
                ]
                if None in shape:
                    raise ModelCompatibilityError(f"Unknown cache dimensions {item.name}: {item.shape}")
                feeds[item.name] = np.zeros(
                    shape, dtype=np.float16 if item.type == "tensor(float16)" else np.float32
                )
            logits = session.run(["logits"], feeds)[0]
            counts = [len(branch.candidate_spans) for branch in branches]
        return np.split(np.asarray(logits).reshape(-1), np.cumsum(counts)[:-1])

    def predict(self, body, protocol=None, *, media_root=".", execution=None, temperature=1.0):
        with self.lock:
            if self.closed:
                raise RuntimeError("Valen client is closed")
            if not isinstance(body, dict):
                raise ValueError("Request must be an object")
            body = body.get("request", body)
            if not isinstance(body, dict):
                raise ValueError("Request must be an object")
            if body.get("model", self.name) != self.name:
                raise ValueError(f"Unknown model; this instance serves {self.name!r}")
            public = "input" in body
            request = to_internal(body, allow_video=True) if public else deepcopy(body)
            source = request.get("state")
            if not isinstance(source, (str, dict)) or (
                isinstance(source, dict) and not isinstance(source.get("messages"), list)
            ):
                raise ValueError("state must be text or a messages object")
            if not isinstance(request.get("questions"), dict) or not request["questions"]:
                raise ValueError("questions must be a nonempty object")
            for question in request["questions"].values():
                if not isinstance(question, dict):
                    raise ValueError("questions must contain question objects")
                candidates(question)
            temperature = body.get("temperature", temperature)
            if isinstance(temperature, dict):
                for kind in {question["type"] for question in request["questions"].values()}:
                    value = temperature.get(kind)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        raise ValueError(f"temperature[{kind!r}] must be finite and positive")
            if not isinstance(temperature, dict) and (
                isinstance(temperature, bool)
                or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature)
                or temperature <= 0
            ):
                raise ValueError("temperature must be finite and positive")
            with local_videos(request) as videos, local_images(videos) as materialized:
                state = Compiler(
                    self.processor,
                    media_root,
                    self.max_length,
                    self.media_kwargs,
                    execution=execution or body.get("execution", self.execution),
                ).compile({"request": materialized})
                if state.inputs is not None:
                    branches = [branch for question in state.questions for branch in question.branches]
                    values = self.run(state.inputs, branches)
                else:
                    values = [
                        self.run(branch.inputs, [branch])[0]
                        for question in state.questions
                        for branch in question.branches
                    ]
                iterator = iter(values)
                logits = [
                    np.concatenate([next(iterator) for _ in question.branches])
                    for question in state.questions
                ]
                response = {
                    "model": self.name,
                    "answers": {
                        question.qid: answer(
                            question,
                            z,
                            temperature[question.kind] if isinstance(temperature, dict) else temperature,
                        )
                        for question, z in zip(state.questions, logits)
                    },
                    "usage": {"input_tokens": state.logical_tokens, "output_tokens": 0},
                    "internal_usage": {"compute_tokens": state.compute_tokens},
                }
                return to_public(body, response) if public else response

    def close(self):
        with self.lock:
            self.closed = True
            self.sessions.clear()
            self.processor = None
            gc.collect()
