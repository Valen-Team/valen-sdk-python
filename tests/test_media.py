import base64
from pathlib import Path
import pytest
from valen_sdk.media import local_videos


def request(data):
    return {'state': {'messages': [{'role': 'user', 'content': [{'type': 'video_url', 'video_url': {'url': data}}]}]}}


def test_inline_video_materialization_and_cleanup_on_failure():
    body = request('data:video/mp4;base64,' + base64.b64encode(b'fixture').decode())
    path = None
    with pytest.raises(RuntimeError):
        with local_videos(body) as actual:
            path = Path(actual['state']['messages'][0]['content'][0]['video_url']['url'])
            assert path.read_bytes() == b'fixture'
            assert body['state']['messages'][0]['content'][0]['video_url']['url'].startswith('data:')
            raise RuntimeError('inference failed')
    assert path is not None and not path.exists() and not path.parent.exists()


@pytest.mark.parametrize('data', ['data:video/mp4;base64,!!', 'data:video/mp4;base64,', 'data:video/unknown;base64,YQ==', None])
def test_inline_video_validation(data):
    with pytest.raises(ValueError):
        with local_videos(request(data)):
            pass


def test_inline_video_request_budget():
    body = request('data:video/mp4;base64,' + base64.b64encode(b'12345').decode())
    with pytest.raises(ValueError, match='limit'):
        with local_videos(body, max_bytes=4):
            pass
