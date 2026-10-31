import json
from pathlib import Path
import sys
import types
import pytest
from valen_sdk.assets import resolve_directory
from valen_sdk.errors import ModelCompatibilityError


def test_selective_fixed_revision_and_offline_cache(tmp_path, monkeypatch):
    manifest = tmp_path / 'valen_onnx.json'
    manifest.write_text(json.dumps({'variants': {'fp32': {'files': ['config.json', 'fp32/decision.onnx']},
                                                'q8': {'files': ['config.json', 'q8/decision.onnx']}}}))
    calls = []
    def download(repo, filename, **kwargs):
        calls.append(('manifest', repo, filename, kwargs))
        return str(manifest)
    def snapshot(repo, **kwargs):
        for name in ["config.json", "q8/decision.onnx"]:
            asset = tmp_path / name
            asset.parent.mkdir(parents=True, exist_ok=True)
            asset.touch()
        calls.append(('snapshot', repo, kwargs))
        return str(tmp_path)
    hub = types.ModuleType('huggingface_hub')
    hub.hf_hub_download, hub.snapshot_download = download, snapshot
    errors = types.ModuleType('huggingface_hub.errors')
    errors.EntryNotFoundError = type('EntryNotFoundError', (Exception,), {})
    errors.LocalEntryNotFoundError = type('LocalEntryNotFoundError', (Exception,), {})
    monkeypatch.setitem(sys.modules, 'huggingface_hub', hub)
    monkeypatch.setitem(sys.modules, 'huggingface_hub.errors', errors)
    assert resolve_directory('owner/model', variant='q8', revision='fixed-sha', local_files_only=True,
                             cache_dir='cache') == tmp_path
    assert calls[1][2]['allow_patterns'] == ['valen_onnx.json', 'config.json', 'q8/decision.onnx']
    for call in calls:
        options = call[-1]
        assert options['revision'] == 'fixed-sha' and options['local_files_only'] is True
    with pytest.raises(ModelCompatibilityError, match='Unavailable variant'):
        resolve_directory('owner/model', variant='unavailable')
    manifest.write_text(json.dumps({'variants': {'q8': {'files': ['../outside.data']}}}))
    with pytest.raises(ModelCompatibilityError, match='Invalid model asset path'):
        resolve_directory('owner/model', variant='q8')


def test_existing_read_only_directory_and_missing_explicit_path(tmp_path):
    assert resolve_directory(tmp_path, local_files_only=True) == tmp_path
    with pytest.raises(ModelCompatibilityError, match='does not exist'):
        resolve_directory(tmp_path / 'missing')
