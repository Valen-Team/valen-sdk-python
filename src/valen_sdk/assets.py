"""Selective model downloads through the mature Hugging Face cache."""

import json
from pathlib import Path
from .errors import ModelCompatibilityError
from .manifest import validate_asset_path


def resolve_directory(model, *, revision=None, local_files_only=False, cache_dir=None, variant="fp32"):
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    if isinstance(model, Path) or path.is_absolute() or str(model).startswith(("./", "../", "~")):
        raise ModelCompatibilityError(f"Local model directory does not exist: {path}")
    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    options = dict(revision=revision, local_files_only=local_files_only, cache_dir=cache_dir)
    try:
        manifest_path = hf_hub_download(str(model), "valen_onnx.json", **options)
    except (EntryNotFoundError, LocalEntryNotFoundError):
        if variant != "fp32":
            raise ModelCompatibilityError("Legacy bundles expose only their original FP32 variant")
        patterns = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "processor_config.json",
            "chat_template.jinja",
            "onnx/*.onnx",
            "onnx/*.data",
        ]
    else:
        manifest = json.loads(Path(manifest_path).read_text())
        if variant not in manifest["variants"]:
            raise ModelCompatibilityError(
                f"Unavailable variant {variant!r}; choose {list(manifest['variants'])}"
            )
        patterns = [
            "valen_onnx.json",
            *[validate_asset_path(name) for name in manifest["variants"][variant]["files"]],
        ]
    return Path(snapshot_download(str(model), allow_patterns=patterns, **options))
