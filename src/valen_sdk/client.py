"""One business interface with local ONNX, remote HTTP and LLM comparison backends."""

import asyncio
import time
from pydantic import BaseModel
import httpx

from . import __version__
from .errors import ValenAPIError
from .types import DecisionResponse, SystemOneResponse

RETRYABLE = {429, 502, 503, 504}


def _headers(api_key, headers):
    return {"User-Agent": "valen-sdk-python/" + __version__,
            **({"Authorization": "Bearer " + api_key} if api_key else {}), **(headers or {})}


def _backoff(attempt):
    return min(0.25 * 2**attempt, 2)


def _result(response):
    """The JSON body, or ValenAPIError with the status, request ID and the service's detail."""
    # Self-hosted servers send x-request-id; the hosted APIs send x-llm-proxy-request-id.
    request_id = response.headers.get("x-request-id") or response.headers.get("x-llm-proxy-request-id")
    try:
        result = response.json()
    except ValueError as exc:
        raise ValenAPIError(
            f"Valen returned HTTP {response.status_code} without JSON: {response.text[:200]}",
            status_code=response.status_code,
            request_id=request_id,
        ) from exc
    if response.is_error:
        detail = result.get("error", result.get("detail", "Valen request failed")) if isinstance(result, dict) else result
        if isinstance(detail, dict):
            detail = detail.get("message", detail)
        raise ValenAPIError(str(detail), status_code=response.status_code, request_id=request_id)
    return result


def _body(state, questions, options):
    return {
        "state": state,
        "questions": {
            key: question.model_dump() if isinstance(question, BaseModel) else question
            for key, question in questions.items()
        },
        **options,
    }


class Valen:
    def __init__(self, backend):
        self._backend = backend

    @classmethod
    def local(cls, model, **options):
        from .onnx import ONNXBackend

        return cls(ONNXBackend(model, **options))

    @classmethod
    def remote(cls, base_url, *, api_key=None, model=None, timeout=120.0, max_retries=2, **options):
        return cls(
            HTTPBackend(
                base_url, api_key=api_key, model=model, timeout=timeout, max_retries=max_retries, **options
            )
        )

    @classmethod
    def llm(cls, provider, model, **options):
        """Answer with an OpenAI-compatible, Anthropic or Gemini model, for side-by-side comparison."""
        from .llm import LLMBackend

        return cls(LLMBackend(provider, model, **options))

    def decisions(self, *, input, questions, **options):
        return DecisionResponse.model_validate(
            self._backend.predict({"input": input, "questions": questions, **options}, "decisions")
        )

    def system_one(self, state, questions, **options):
        return SystemOneResponse.model_validate(
            self._backend.predict(_body(state, questions, options), "system_one")
        )

    def predict(self, request, **options):
        """Raw JSON interface, including wrapped legacy requests."""
        if not isinstance(request, dict):
            raise ValueError("Request must be an object")
        body = request.get("request", request)
        if not isinstance(body, dict):
            raise ValueError("Request must be an object")
        return self._backend.predict(body, "decisions" if "input" in body else "system_one", **options)

    def close(self):
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class HTTPBackend:
    def __init__(
        self,
        base_url,
        *,
        api_key=None,
        model=None,
        timeout=120.0,
        max_retries=2,
        transport=None,
        headers=None,
    ):
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.model, self.max_retries = model, max_retries
        self.closed = False
        self.client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers=_headers(api_key, headers),
            transport=transport,
        )

    def predict(self, body, protocol, **options):
        if self.closed:
            raise RuntimeError("Valen client is closed")
        body = {**({"model": self.model} if self.model else {}), **body, **options}
        path = "decisions" if protocol == "decisions" else "systemone"
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(path, json=body)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise ValenAPIError("Unable to reach Valen service") from exc
                time.sleep(_backoff(attempt))
                continue
            if response.status_code in RETRYABLE and attempt < self.max_retries:
                time.sleep(_backoff(attempt))
                continue
            return _result(response)

    def close(self):
        self.closed = True
        self.client.close()


class AsyncValen:
    """Asynchronous HTTP client; local inference uses the synchronous Valen API."""

    def __init__(
        self, base_url, *, api_key=None, model=None, timeout=120.0, max_retries=2, transport=None, headers=None
    ):
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.model, self.max_retries = model, max_retries
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers=_headers(api_key, headers),
            transport=transport,
        )

    async def _post(self, path, body):
        body = {**({"model": self.model} if self.model else {}), **body}
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(path, json=body)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise ValenAPIError("Unable to reach Valen service") from exc
                await asyncio.sleep(_backoff(attempt))
                continue
            if response.status_code in RETRYABLE and attempt < self.max_retries:
                await asyncio.sleep(_backoff(attempt))
                continue
            return _result(response)

    async def decisions(self, *, input, questions, **options):
        return DecisionResponse.model_validate(
            await self._post("decisions", {"input": input, "questions": questions, **options})
        )

    async def system_one(self, state, questions, **options):
        return SystemOneResponse.model_validate(
            await self._post("systemone", _body(state, questions, options))
        )

    async def close(self):
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()
