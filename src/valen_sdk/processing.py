"""Framework-free Qwen preprocessing using HF tokenizers, Jinja and Pillow.

Resize/patch layouts and mRoPE are adapted from Transformers Qwen2-VL PIL,
Qwen3-VL and Qwen3.5 implementations (Apache-2.0). See THIRD_PARTY_NOTICES.md.
"""

from copy import deepcopy
import itertools
import json
import math
import re
from pathlib import Path

import numpy as np


class Tokenizer:
    def __init__(self, directory):
        from tokenizers import Tokenizer as Backend

        path = Path(directory) / "tokenizer.json"
        self.backend = Backend.from_file(str(path))
        data = json.loads(path.read_text())
        self.all_special_tokens = [x["content"] for x in data.get("added_tokens", []) if x.get("special")]

    def encode(self, text, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = self.backend.encode(text, add_special_tokens=add_special_tokens)
        output = {"input_ids": result.ids}
        if return_offsets_mapping:
            output["offset_mapping"] = result.offsets
        return output


def smart_resize(height, width, factor, minimum, maximum, frames=None, temporal=2):
    if height <= 0 or width <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError("Image dimensions must be positive with aspect ratio at most 200")
    if frames is not None and frames < temporal:
        raise ValueError("Video must contain at least one temporal patch of sampled frames")
    if frames is not None and (height < factor or width < factor):
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    h, w = round(height / factor) * factor, round(width / factor) * factor
    count = 1 if frames is None else round(frames / temporal) * temporal
    if count * h * w > maximum:
        beta = math.sqrt((height * width * (frames or 1)) / maximum)
        h, w = (
            max(factor, math.floor(height / beta / factor) * factor),
            max(factor, math.floor(width / beta / factor) * factor),
        )
    elif count * h * w < minimum:
        beta = math.sqrt(minimum / (height * width * (frames or 1)))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    return h, w


def sample_frame_indices(total_frames, fps, config, default_fps=2):
    if total_frames < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid video frame metadata")
    count = config.get("num_frames")
    requested_fps = config.get("fps")
    if config.get("do_sample_frames", True) is False:
        count = total_frames
    else:
        if count is not None and requested_fps is not None:
            raise ValueError("num_frames and fps are mutually exclusive")
        if count is None:
            requested_fps = requested_fps if requested_fps is not None else default_fps
            if requested_fps is not None:
                if (
                    isinstance(requested_fps, bool)
                    or not isinstance(requested_fps, (int, float))
                    or not math.isfinite(requested_fps)
                    or requested_fps <= 0
                ):
                    raise ValueError("Video fps must be finite and positive")
                count = min(
                    max(int(total_frames / fps * requested_fps), config.get("min_frames", 4)),
                    config.get("max_frames", 768),
                    total_frames,
                )
            else:
                count = min(max(total_frames, config.get("min_frames", 4)), config.get("max_frames", 768))
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("num_frames must be a positive integer")
    return np.linspace(0, total_frames - 1, count).round().astype(int)


def image_patches(images, config, *, video=False):
    from PIL import Image

    patch, merge, temporal = config["patch_size"], config["merge_size"], config["temporal_patch_size"]
    height, width = images[0].height, images[0].width
    if config.get("do_resize", True):
        maximum = config["size"]["longest_edge"]
        if video and config.get("cap_pixels_per_frame", False):
            cap = config.get("max_video_tokens", 768) * (patch * merge) ** 2
            per_frame = max(min(cap, maximum // len(images)), int(config["size"]["shortest_edge"] * 1.05))
            maximum = per_frame * len(images)
        height, width = smart_resize(
            height,
            width,
            patch * merge,
            config["size"]["shortest_edge"],
            maximum,
            frames=len(images) if video else None,
            temporal=temporal,
        )
    mean = np.asarray(config["image_mean"], dtype=np.float32)[None, None, :]
    std = np.asarray(config["image_std"], dtype=np.float32)[None, None, :]
    frames = []
    for image in images:
        if config.get("do_convert_rgb", True):
            image = image.convert("RGB")
        image = image.resize((width, height), resample=Image.Resampling(config.get("resample", 3)))
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("Valen vision requires three image channels")
        if config.get("do_rescale", True):
            array = (array.astype(np.float64) * config.get("rescale_factor", 1 / 255)).astype(np.float32)
        else:
            array = array.astype(np.float32)
        if config.get("do_normalize", True):
            array = (array - mean) / std
        frames.append(array.transpose(2, 0, 1))
    if not video:
        frames *= temporal
    while len(frames) % temporal:
        frames.append(frames[-1])
    t, h, w = len(frames) // temporal, height // patch, width // patch
    pixels = np.stack(frames).reshape(t, temporal, 3, h // merge, merge, patch, w // merge, merge, patch)
    pixels = pixels.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(t * h * w, 3 * temporal * patch * patch)
    return pixels.astype(np.float32), np.asarray([t, h, w], dtype=np.int64)


def decode_video_frames(path, config, fallback_fps):
    try:
        import av
    except ImportError as exc:
        raise ImportError("Video inputs require valen-sdk[video]") from exc
    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise ValueError("Video has no video stream")
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 24.0
            total = stream.frames or sum(1 for _ in container.decode(stream))
            width, height = stream.width, stream.height
        if not total:
            raise ValueError("Video contains no decodable frames")
        indices = sample_frame_indices(total, fps, config, fallback_fps)
        if width * height * 3 * len(indices) > 512 * 1024 * 1024:
            raise ValueError("Selected video frames exceed the 512 MiB decoded-memory limit")
        selected, wanted = {}, set(indices.tolist())
        with av.open(path) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in wanted:
                    selected[index] = frame.to_image()
                if index >= int(indices.max()):
                    break
        if set(selected) != wanted:
            raise ValueError("Video metadata does not match its decodable frame count")
        return [selected[i] for i in indices], indices, fps, total
    except av.error.FFmpegError as exc:
        raise ValueError("Video data could not be decoded") from exc


class Processor:
    def __init__(self, directory):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        directory = Path(directory)
        self.tokenizer = Tokenizer(directory)
        self.config = json.loads((directory / "processor_config.json").read_text())
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )

        def raise_exception(message):
            raise ValueError(message)

        environment.globals["raise_exception"] = raise_exception
        self.template = environment.from_string((directory / "chat_template.jinja").read_text())

    def apply_chat_template(self, messages, *, processor_kwargs=None, **kwargs):
        from PIL import Image

        options = processor_kwargs or {}
        text = self.template.render(messages=messages, add_generation_prompt=False)
        output, image_grids, video_grids, image_pixels, video_pixels, metadata = {}, [], [], [], [], []
        # Replacement callbacks preserve the original template and media ordering.
        for message in messages:
            for item in message["content"]:
                if item["type"] == "image":
                    config = deepcopy(self.config["image_processor"])
                    config.update(options.get("images_kwargs", {}))
                    try:
                        with Image.open(item["path"]) as image:
                            pixels, grid = image_patches([image], config)
                    except OSError as exc:
                        raise ValueError("Image data could not be decoded") from exc
                    image_pixels.append(pixels)
                    image_grids.append(grid)
                elif item["type"] == "video":
                    config = deepcopy(self.config["video_processor"])
                    config.update(options.get("videos_kwargs", {}))
                    frames, indices, fps, total_frames = decode_video_frames(
                        item["path"], config, self.config["video_processor"].get("fps")
                    )
                    pixels, grid = image_patches(frames, config, video=True)
                    video_pixels.append(pixels)
                    video_grids.append(grid)
                    padded = indices.tolist()
                    while len(padded) % config["temporal_patch_size"]:
                        padded.append(padded[-1])
                    timestamps = [
                        (padded[i] + padded[i + config["temporal_patch_size"] - 1]) / (2 * fps)
                        for i in range(0, len(padded), config["temporal_patch_size"])
                    ]
                    metadata.append(
                        {"fps": fps, "total_num_frames": total_frames, "frames_indices": indices.tolist()}
                    )
                    item["_timestamps"] = timestamps
        image_iterator = iter(image_grids)

        def replace_image(match):
            grid = next(image_iterator)
            count = int(grid.prod()) // self.config["image_processor"]["merge_size"] ** 2
            return "<|image_pad|>" * count

        text = re.sub(r"<\|image_pad\|>", replace_image, text)
        video_index = 0
        for message in messages:
            for item in message["content"]:
                if item["type"] == "video":
                    grid = video_grids[video_index]
                    count = int(grid[1] * grid[2]) // self.config["video_processor"]["merge_size"] ** 2
                    replacement = "".join(
                        f"<{timestamp:.1f} seconds><|vision_start|>"
                        + "<|video_pad|>" * count
                        + "<|vision_end|>"
                        for timestamp in item.pop("_timestamps")
                    )
                    # The trained processor keeps the template's outer envelope
                    # around its timestamped per-frame envelopes.
                    text = text.replace(
                        "<|vision_start|><|video_pad|><|vision_end|>",
                        "<|vision_start|>" + replacement + "<|vision_end|>",
                        1,
                    )
                    video_index += 1
        ids = np.asarray([self.tokenizer.encode(text)], dtype=np.int64)
        output.update(input_ids=ids, attention_mask=np.ones_like(ids), mm_token_type_ids=np.zeros_like(ids))
        for token, value in [("<|image_pad|>", 1), ("<|video_pad|>", 2)]:
            output["mm_token_type_ids"][ids == self.tokenizer.backend.token_to_id(token)] = value
        if image_grids:
            output.update(pixel_values=np.concatenate(image_pixels), image_grid_thw=np.stack(image_grids))
        if video_grids:
            output.update(
                pixel_values_videos=np.concatenate(video_pixels),
                video_grid_thw=np.stack(video_grids),
                video_metadata=metadata,
            )
        return output


def position_ids(inputs, merge):
    """NumPy adaptation of Qwen3.5 get_rope_index, for unpadded single states."""
    ids, types = inputs["input_ids"], inputs.get("mm_token_type_ids", np.zeros_like(inputs["input_ids"]))
    grids = {}
    grids[1] = iter(inputs.get("image_grid_thw", []))
    video = inputs.get("video_grid_thw", [])
    split_video = [np.asarray([1, h, w]) for t, h, w in video for _ in range(t)]
    grids[2] = iter(split_video)
    positions, current = [], 0
    for kind, group in itertools.groupby(types[0].tolist()):
        count = sum(1 for _ in group)
        if kind == 0:
            positions.append(np.broadcast_to(np.arange(count), (3, count)) + current)
            current += count
        else:
            t, h, w = next(grids[kind])
            grid = np.stack(
                np.meshgrid(np.arange(t), np.arange(h // merge), np.arange(w // merge), indexing="ij")
            )
            grid = grid.reshape(3, -1) + current
            if grid.shape[1] != count:
                raise ValueError("Visual placeholders and grid do not agree")
            positions.append(grid)
            current += max(h, w) // merge
    return np.concatenate(positions, axis=1)[:, None, :].astype(np.int64)
