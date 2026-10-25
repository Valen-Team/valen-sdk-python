import pytest
from valen_sdk.errors import ModelCompatibilityError
from valen_sdk.manifest import validate_asset_path, validate_variant


def fixture():
    return {'format_version': 2, 'contract_version': 'valen-decisions-1', 'variants': {'fp32': {
        'minimum_onnxruntime': '1.31.0', 'graphs': {'decision': 'onnx/decision.onnx'}, 'files': ['config.json']}}}


def test_manifest_versions_and_paths():
    manifest = fixture()
    assert validate_variant(manifest, 'fp32', '1.31.0') is manifest['variants']['fp32']
    with pytest.raises(ModelCompatibilityError, match='requires ONNX Runtime'):
        validate_variant(manifest, 'fp32', '1.30.0')
    with pytest.raises(ModelCompatibilityError, match='Unavailable variant'):
        validate_variant(manifest, 'q4', '1.31.0')
    manifest['contract_version'] = 'future'
    with pytest.raises(ModelCompatibilityError, match='decision contract'):
        validate_variant(manifest, 'fp32', '1.31.0')
    for name in ['../weights', '/weights', 'a/../b', 'a\\b', 'a//b', 'a\0b']:
        with pytest.raises(ModelCompatibilityError, match='Invalid model asset path'):
            validate_asset_path(name)
