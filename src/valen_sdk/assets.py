"""Selective model downloads through the mature Hugging Face cache."""

import json
from pathlib import Path
from .errors import ModelCompatibilityError
from .manifest import validate_asset_path


def resolve_directory(model, *, revision=None, local_files_only=False, cache_dir=None, variant="fp32", on_event=None):
    from .client import _emit
    _emit(on_event, event="resolve_start", backend="local", variant=variant)
    path = Path(model).expanduser()
    if path.is_dir():
        _emit(on_event, event="resolve_complete", backend="local", variant=variant, cached=True)
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
    _emit(on_event, event="download_start", backend="local", variant=variant)
    try:
        directory = Path(snapshot_download(str(model), allow_patterns=patterns, **options))
    except LocalEntryNotFoundError as exc:
        raise ModelCompatibilityError(
            f"Offline cache is incomplete for {model!r}, revision={revision!r}, variant={variant!r}; "
            "download this revision and variant online first"
        ) from exc
    missing = [name for name in patterns if "*" not in name and not (directory / name).is_file()]
    if missing:
        raise ModelCompatibilityError("Missing model assets: " + ", ".join(missing) + "; download the selected revision and variant again")
    _emit(on_event, event="resolve_complete", backend="local", variant=variant)
    return directory
