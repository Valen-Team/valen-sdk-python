"""Materialize explicitly supplied inline videos without fetching network URLs."""

import base64
import binascii
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def local_videos(request, max_bytes=64 * 1024 * 1024):
    materialized = deepcopy(request)
    state = materialized.get("state")
    messages = state.get("messages", []) if isinstance(state, dict) else []
    with TemporaryDirectory(prefix="valen-video-") as folder:
        index = 0
        total = 0
        for message in messages:
            for part in message.get("content", []) if isinstance(message.get("content"), list) else []:
                if part.get("type") != "video_url":
                    continue
                value = part.get("video_url")
                url = value.get("url") if isinstance(value, dict) else value
                if not isinstance(url, str):
                    raise ValueError("video_url must contain a URL string")
                if isinstance(value, str):
                    part["video_url"] = {"url": url}
                if not url.startswith("data:"):
                    continue
                header, separator, payload = url.partition(",")
                extensions = {
                    "data:video/mp4;base64": ".mp4",
                    "data:video/webm;base64": ".webm",
                    "data:video/ogg;base64": ".ogv",
                    "data:video/quicktime;base64": ".mov",
                }
                if not separator or header not in extensions:
                    raise ValueError("Inline videos require a supported base64 video data URL")
                if len(payload) > ((max_bytes - total + 2) // 3) * 4:
                    raise ValueError("Inline video data exceeds the 64 MiB request limit")
                try:
                    data = base64.b64decode(payload, validate=True)
                except (ValueError, binascii.Error) as exc:
                    raise ValueError("Invalid base64 video data") from exc
                total += len(data)
                if not data or total > max_bytes:
                    raise ValueError("Inline video data is empty or exceeds the request limit")
                path = Path(folder) / (str(index) + extensions[header])
                path.write_bytes(data)
                part["video_url"] = {"url": str(path)}
                index += 1
        yield materialized
