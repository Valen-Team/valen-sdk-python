"""One business interface with local ONNX, remote HTTP and LLM comparison backends."""

import time
from pydantic import BaseModel
import httpx

from .errors import ValenAPIError
from .types import DecisionResponse, SystemOneResponse


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
            headers={**({"Authorization": "Bearer " + api_key} if api_key else {}), **(headers or {})},
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
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                if attempt == self.max_retries:
                    raise ValenAPIError("Unable to reach Valen service") from exc
                time.sleep(min(0.25 * 2**attempt, 2))
                continue
            if response.status_code in {429, 502, 503, 504} and attempt < self.max_retries:
                time.sleep(min(0.25 * 2**attempt, 2))
                continue
            if response.is_error:
                try:
                    detail = response.json().get(
                        "error", response.json().get("detail", "Valen request failed")
                    )
                except ValueError:
                    detail = "Valen request failed"
                raise ValenAPIError(
                    str(detail),
                    status_code=response.status_code,
                    request_id=response.headers.get("x-request-id"),
                )
            return response.json()

    def close(self):
        self.closed = True
        self.client.close()


class AsyncValen:
    """Asynchronous HTTP client; local inference uses the synchronous Valen API."""

    def __init__(self, base_url, *, api_key=None, model=None, timeout=120.0, transport=None):
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            headers={"Authorization": "Bearer " + api_key} if api_key else {},
            transport=transport,
        )

    async def _post(self, path, body):
        try:
            response = await self.client.post(
                path, json={**({"model": self.model} if self.model else {}), **body}
            )
        except httpx.HTTPError as exc:
            raise ValenAPIError("Unable to reach Valen service") from exc
        if response.is_error:
            raise ValenAPIError(
                "Valen request failed",
                status_code=response.status_code,
                request_id=response.headers.get("x-request-id"),
            )
        return response.json()

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
