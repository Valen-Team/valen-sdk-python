"""Validate the portable graph contract before opening any model graph."""

from packaging.version import Version
from .errors import ModelCompatibilityError


def validate_asset_path(name):
    if (
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or "\\" in name
        or "\0" in name
        or ":" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ModelCompatibilityError(f"Invalid model asset path: {name!r}")
    return name


def validate_variant(manifest, variant, runtime_version):
    if manifest.get("format_version") != 2:
        raise ModelCompatibilityError("Unsupported Valen ONNX manifest version")
    if manifest.get("contract_version") != "valen-decisions-1":
        raise ModelCompatibilityError("Unsupported Valen decision contract")
    variants = manifest["variants"]
    if variant not in variants:
        raise ModelCompatibilityError(f"Unavailable variant {variant!r}")
    selected = variants[variant]
    for name in {*selected["files"], *selected["graphs"].values(), *selected.get("sha256", {})}:
        validate_asset_path(name)
    required = selected.get("minimum_onnxruntime")
    if required and Version(runtime_version) < Version(required):
        raise ModelCompatibilityError(
            f"This model requires ONNX Runtime >= {required}; installed {runtime_version}"
        )
    return selected
