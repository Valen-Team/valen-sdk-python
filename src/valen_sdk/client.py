"""One business interface with local ONNX, remote HTTP and LLM comparison backends."""

import asyncio
import time
import random
import math
import warnings
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from pydantic import BaseModel
import httpx

from . import __version__
from .errors import ValenAPIError, ValenTimeoutError, ValenResponseError, ValenInputError
from .validation import validate_response, validate_request
from pydantic import ValidationError
from .types import DecisionResponse, SystemOneResponse

RETRYABLE = {429, 502, 503, 504}


def _headers(api_key, headers):
    return {"User-Agent": "valen-sdk-python/" + __version__,
            **({"Authorization": "Bearer " + api_key} if api_key else {}), **(headers or {})}


def _backoff(attempt):
    return random.uniform(0, min(0.25 * 2**attempt, 2))


def _positive(value, name):
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0):
        raise ValueError(name + " must be finite and positive")
    return value


def _remaining(deadline):
    remaining = None if deadline is None else deadline - time.monotonic()
    if remaining is not None and remaining <= 0:
        raise ValenTimeoutError()
    return remaining


def _delay(attempt, response=None):
    value = response.headers.get("retry-after") if response is not None else None
    if value:
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = float("nan")
        if math.isfinite(seconds):
            return max(0, seconds)
    return _backoff(attempt)


def _emit(callback, **event):
    if callback:
        try:
            callback(event)
        except Exception:
            warnings.warn("Valen on_event callback failed", RuntimeWarning, stacklevel=2)


def _validated(body, result, cls):
    try:
        validate_response(body, result)
    except ValenResponseError as exc:
        exc.request_id = getattr(result, "request_id", None)
        exc.status_code = getattr(result, "status_code", None)
        raise
    try:
        return cls.model_validate(result)
    except ValidationError as exc:
        field = ".".join(map(str, exc.errors()[0]["loc"])) or "response"
        raise ValenResponseError("Response schema validation failed at " + field, field=field, model=result.get("model"),
                                 request_id=getattr(result, "request_id", None), status_code=getattr(result, "status_code", None),
                                 debug={"validation_errors": exc.errors(include_input=False)}) from exc


class _ResponseBody(dict):
    """Keep transport metadata without changing the public JSON keys."""
    def __init__(self, body, response):
        super().__init__(body)
        self.request_id = response.headers.get("x-request-id") or response.headers.get("x-llm-proxy-request-id")
        self.status_code = response.status_code


def _result(response):
    """The JSON body, or ValenAPIError with the status, request ID and the service's detail."""
    # Self-hosted servers send x-request-id; the hosted APIs send x-llm-proxy-request-id.
    request_id = response.headers.get("x-request-id") or response.headers.get("x-llm-proxy-request-id")
    try:
        result = response.json()
    except ValueError as exc:
        error_type = ValenResponseError if response.is_success else ValenAPIError
        raise error_type(
            f"Valen returned HTTP {response.status_code} without JSON: {response.text[:200]}",
            status_code=response.status_code,
            request_id=request_id,
        ) from exc
    if response.is_error:
        detail = result.get("error", result.get("detail", "Valen request failed")) if isinstance(result, dict) else result
        if isinstance(detail, dict):
            detail = detail.get("message", detail)
        raise ValenAPIError(str(detail), status_code=response.status_code, request_id=request_id)
    return _ResponseBody(result, response) if isinstance(result, dict) else result


def _body(state, questions, options):
    if not isinstance(questions, dict):
        raise ValenInputError("questions: expected an object", field="questions")
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
        body = {"input": input, "questions": questions, **options}
        validate_request(body)
        return _validated(body, self._backend.predict(body, "decisions"), DecisionResponse)

    def system_one(self, state, questions, **options):
        body = _body(state, questions, options)
        validate_request(body)
        return _validated(body, self._backend.predict(body, "system_one"), SystemOneResponse)

    def predict(self, request, **options):
        """Raw JSON interface, including wrapped legacy requests."""
        if not isinstance(request, dict):
            raise ValueError("Request must be an object")
        body = request.get("request", request)
        if not isinstance(body, dict):
            raise ValueError("Request must be an object")
        return self._backend.predict(body, "decisions" if "input" in body else "system_one", **options)

    def warmup(self, request):
        """Run a caller-supplied representative request, loading lazy local sessions."""
        return self.predict(request)

    @property
    def info(self):
        return getattr(self._backend, "info", {"backend": type(self._backend).__name__})

    def close(self):
        self._backend.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class HTTPBackend:
    def __init__(self, base_url, *, api_key=None, model=None, timeout=120.0, max_retries=2,
                 transport=None, headers=None, total_timeout=None, http_client=None,
                 on_event=None):
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.timeout = _positive(timeout, "timeout")
        self.total_timeout = _positive(total_timeout, "total_timeout")
        self.model, self.max_retries = model, max_retries
        self.closed = False
        self.on_event = on_event
        self.base_url = base_url.rstrip("/") + "/"
        self.headers = _headers(api_key, headers)
        if http_client is not None and transport is not None:
            raise ValueError("Pass http_client or transport, not both")
        self.owns_client = http_client is None
        self.client = http_client if http_client is not None else httpx.Client(transport=transport)

    def predict(self, body, protocol, **options):
        if self.closed:
            raise RuntimeError("Valen client is closed")
        body = {**({"model": self.model} if self.model else {}), **body, **options}
        timeout = _positive(body.pop("timeout", self.timeout), "timeout")
        total = _positive(body.pop("total_timeout", self.total_timeout), "total_timeout")
        started = time.monotonic()
        deadline = None if total is None else started + total
        path = "decisions" if protocol == "decisions" else "systemone"
        _emit(self.on_event, event="start", backend="http", model=body.get("model"))
        try:
            for attempt in range(self.max_retries + 1):
                remaining = _remaining(deadline)
                per_attempt = min(timeout, remaining) if remaining is not None and timeout is not None else (remaining or timeout)
                response = None
                try:
                    response = self.client.post(self.base_url + path, json=body, headers=self.headers, timeout=per_attempt)
                except httpx.TransportError as exc:
                    _emit(self.on_event, event="attempt", attempt=attempt+1, code=type(exc).__name__)
                    _remaining(deadline)
                    if attempt == self.max_retries:
                        raise ValenAPIError("Unable to reach Valen service", code="timeout" if isinstance(exc, httpx.TimeoutException) else "network_error",
                                            model=body.get("model"), retryable=True) from exc
                else:
                    _remaining(deadline)
                    _emit(self.on_event, event="attempt", attempt=attempt+1, status_code=response.status_code,
                          request_id=response.headers.get("x-request-id") or response.headers.get("x-llm-proxy-request-id"))
                    if response.status_code not in RETRYABLE or attempt == self.max_retries:
                        result = _result(response)
                        _emit(self.on_event, event="complete", attempts=attempt+1, elapsed=time.monotonic()-started,
                              model=result.get("model") if isinstance(result, dict) else None,
                              usage=result.get("usage") if isinstance(result, dict) else None)
                        return result
                delay = _delay(attempt, response)
                remaining = _remaining(deadline)
                if remaining is not None and delay >= remaining:
                    raise ValenTimeoutError(model=body.get("model"))
                _emit(self.on_event, event="retry", attempt=attempt+1, delay=delay)
                time.sleep(delay)
        except Exception as exc:
            _emit(self.on_event, event="error", code=getattr(exc, "code", "request_error"), elapsed=time.monotonic()-started)
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            if self.owns_client:
                self.client.close()


class AsyncValen:
    """Asynchronous HTTP client. AsyncValen.llm supports concurrent LLM comparisons."""

    def __init__(self, base_url, *, api_key=None, model=None, timeout=120.0, max_retries=2,
                 transport=None, headers=None, total_timeout=None, http_client=None, on_event=None):
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self.timeout = _positive(timeout, "timeout")
        self.total_timeout = _positive(total_timeout, "total_timeout")
        self.model, self.max_retries = model, max_retries
        self.base_url = base_url.rstrip("/") + "/"
        self.headers = _headers(api_key, headers)
        self.on_event, self.closed = on_event, False
        if http_client is not None and transport is not None:
            raise ValueError("Pass http_client or transport, not both")
        self.owns_client = http_client is None
        self.client = http_client if http_client is not None else httpx.AsyncClient(transport=transport)

    @classmethod
    def llm(cls, provider, model, **options):
        from .llm import AsyncLLMBackend
        return _AsyncLLMValen(AsyncLLMBackend(provider, model, **options))

    async def _post(self, path, body):
        if self.closed:
            raise RuntimeError("Valen client is closed")
        body = {**({"model": self.model} if self.model else {}), **body}
        timeout = _positive(body.pop("timeout", self.timeout), "timeout")
        total = _positive(body.pop("total_timeout", self.total_timeout), "total_timeout")
        started = time.monotonic()
        _emit(self.on_event, event="start", backend="http", model=body.get("model"))
        async def run():
            for attempt in range(self.max_retries + 1):
                response = None
                try:
                    response = await self.client.post(self.base_url + path, json=body, headers=self.headers, timeout=timeout)
                except httpx.TransportError as exc:
                    _emit(self.on_event, event="attempt", attempt=attempt+1, code=type(exc).__name__)
                    if attempt == self.max_retries:
                        raise ValenAPIError("Unable to reach Valen service", code="timeout" if isinstance(exc, httpx.TimeoutException) else "network_error",
                                            model=body.get("model"), retryable=True) from exc
                else:
                    _emit(self.on_event, event="attempt", attempt=attempt+1, status_code=response.status_code,
                          request_id=response.headers.get("x-request-id") or response.headers.get("x-llm-proxy-request-id"))
                    if response.status_code not in RETRYABLE or attempt == self.max_retries:
                        result = _result(response)
                        _emit(self.on_event, event="complete", attempts=attempt+1, elapsed=time.monotonic()-started,
                              model=result.get("model") if isinstance(result, dict) else None,
                              usage=result.get("usage") if isinstance(result, dict) else None)
                        return result
                delay = _delay(attempt, response)
                _emit(self.on_event, event="retry", attempt=attempt+1, delay=delay)
                await asyncio.sleep(delay)
        try:
            return await run() if total is None else await asyncio.wait_for(run(), timeout=total)
        except asyncio.TimeoutError as exc:
            _emit(self.on_event, event="error", code="timeout", elapsed=time.monotonic()-started)
            raise ValenTimeoutError(model=body.get("model")) from exc
        except asyncio.CancelledError:
            _emit(self.on_event, event="error", code="cancelled", elapsed=time.monotonic()-started)
            raise
        except Exception as exc:
            _emit(self.on_event, event="error", code=getattr(exc, "code", "request_error"), elapsed=time.monotonic()-started)
            raise

    async def decisions(self, *, input, questions, **options):
        body = {"input": input, "questions": questions, **options}
        validate_request(body)
        return _validated(body, await self._post("decisions", body), DecisionResponse)

    async def system_one(self, state, questions, **options):
        body = _body(state, questions, options)
        validate_request(body)
        return _validated(body, await self._post("systemone", body), SystemOneResponse)

    async def close(self):
        if not self.closed:
            self.closed = True
            if self.owns_client:
                await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()


class _AsyncLLMValen(AsyncValen):
    def __init__(self, backend):
        self._backend = backend

    async def _post(self, path, body):
        return await self._backend.predict(body, "decisions" if path == "decisions" else "system_one")

    async def close(self):
        await self._backend.close()
