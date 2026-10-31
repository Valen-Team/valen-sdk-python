"""Answer Valen requests with a general LLM, returning the same typed responses."""

import asyncio
import contextvars
import base64
import json
import math
import mimetypes
import os
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import httpx

from .errors import ValenAPIError, ValenTimeoutError
from .client import _positive, _delay, _emit, _remaining
from .media import local_videos
from .protocol import to_internal, to_public
from .responses import from_probabilities
from .schema import candidates

PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta", "GEMINI_API_KEY"),
}
RETRYABLE = {408, 429, 500, 502, 503, 504, 529}
SYSTEM = (
    "You answer typed questions about the evidence. Answer each question on its own, from the evidence only. "
    "A noul question has the candidates true and false. A score question has its levels as candidates, "
    "keyed by zero-based index."
)
PROBABILITIES = " For every question, give a probability for each candidate; the probabilities of one question sum to 1."
DISCRETE = " For every question, give the key of the single best candidate."


class LLMBackend:
    """OpenAI-compatible Chat Completions, Anthropic Messages or Gemini generateContent."""

    def __init__(
        self,
        provider,
        model,
        *,
        api_key=None,
        base_url=None,
        structured=True,
        answer_mode="probabilities",
        max_tokens=None,
        n_retry_malformed=2,
        max_retries=2,
        timeout=120.0,
        video_frames=8,
        extra_body=None,
        transport=None,
        total_timeout=None,
        normalize_probabilities=True,
        on_event=None,
    ):
        if provider not in PROVIDERS:
            raise ValueError("provider must be one of: " + ", ".join(PROVIDERS))
        if answer_mode not in {"probabilities", "discrete"}:
            raise ValueError("answer_mode must be 'probabilities' or 'discrete'")
        for name, value in (("max_retries", max_retries), ("n_retry_malformed", n_retry_malformed)):
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be nonnegative")
        if type(video_frames) is not int or video_frames < 1:
            raise ValueError("video_frames must be a positive integer")
        default_url, key_env = PROVIDERS[provider]
        if api_key is None:
            api_key = os.environ.get(key_env) or (os.environ.get("GOOGLE_API_KEY") if provider == "gemini" else None)
        self.provider, self.model = provider, model
        self.structured, self.answer_mode = structured, answer_mode
        self.max_tokens, self.n_retry_malformed, self.max_retries = max_tokens, n_retry_malformed, max_retries
        self.video_frames, self.extra_body = video_frames, extra_body or {}
        self.total_timeout = _positive(total_timeout, "total_timeout")
        self.timeout = _positive(timeout, "timeout")
        self.normalize_probabilities = normalize_probabilities
        self.on_event = on_event
        self.closed = False
        if provider == "anthropic":
            headers = {"x-api-key": api_key or "", "anthropic-version": "2023-06-01"}
        elif provider == "gemini":
            headers = {"x-goog-api-key": api_key or ""}
        else:
            headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self.client = httpx.Client(
            base_url=(base_url or default_url).rstrip("/") + "/", timeout=timeout, headers=headers, transport=transport
        )

    def predict(self, body, protocol, *, media_root="."):
        total = _positive(body.get("total_timeout", self.total_timeout), "total_timeout")
        token = _deadline.set(None if total is None else time.monotonic()+total)
        history_token = _history.set([])
        started = time.monotonic()
        _emit(self.on_event, event="start", backend="llm", model=self.model)
        try:
            result = self._predict(body, protocol, media_root=media_root)
            _remaining(_deadline.get())
            _emit(self.on_event, event="complete", model=result["model"], usage=result["usage"], elapsed=time.monotonic()-started)
            return result
        except Exception as exc:
            if isinstance(exc, ValenAPIError):
                exc.debug = {**(exc.debug or {}), "llm_attempts": _history.get()}
                exc.model = self.provider + ":" + self.model
            _emit(self.on_event, event="error", code=getattr(exc, "code", "request_error"), elapsed=time.monotonic()-started)
            raise
        finally:
            _deadline.reset(token)
            _history.reset(history_token)

    def _predict(self, body, protocol, *, media_root="."):
        steps = self._prediction(body, protocol, media_root=media_root)
        try:
            pending = next(steps)
            while True:
                pending = steps.send(self._call(*pending))
        except StopIteration as done:
            return done.value
        finally:
            steps.close()

    def _prediction(self, body, protocol, *, media_root="."):
        if self.closed:
            raise RuntimeError("Valen client is closed")
        public = "input" in body
        request = to_internal(body, allow_video=True) if public else body
        questions = {
            qid: SimpleNamespace(
                kind=q["type"],
                instructions=q["instructions"],
                keys=[k for k, _ in candidates(q)],
                descriptions=[d for _, d in candidates(q)],
            )
            for qid, q in request["questions"].items()
        }
        # Question IDs stay out of the prompt, as they do for Valen itself.
        aliases = {"q" + str(i + 1): qid for i, qid in enumerate(questions)}
        listing = {
            alias: {
                "type": questions[qid].kind,
                "instructions": questions[qid].instructions,
                "candidates": dict(zip(questions[qid].keys, questions[qid].descriptions)),
            }
            for alias, qid in aliases.items()
        }
        schema = self._schema({alias: questions[qid].keys for alias, qid in aliases.items()})
        with local_videos(request) as materialized:
            evidence = self._evidence(materialized["state"], Path(media_root))
        conversation = [("user", [*evidence, {"type": "text", "text": "Questions:\n" + json.dumps(listing, ensure_ascii=False, indent=2)}])]
        attempts, retries, malformed = _history.get(), 0, 0
        started = time.perf_counter()
        while True:
            text, retries = yield conversation, schema, attempts, retries
            try:
                raw = _parse_json(text)
                if set(raw) != set(aliases):
                    raise ValueError("answer question names differ from the schema")
                answers = {qid: self._answer(questions[qid], raw.get(alias)) for alias, qid in aliases.items()}
                normalized = {qid: self.answer_mode == "probabilities" and not math.isclose(sum(raw[alias].values()), 1, abs_tol=1e-5, rel_tol=0) for alias, qid in aliases.items()}
                break
            except ValueError as exc:
                attempts[-1]["error"] = str(exc)
                if malformed == self.n_retry_malformed:
                    raise ValenAPIError(
                        "LLM output did not match the answer schema: " + str(exc), debug={"llm_attempts": attempts}
                    ) from None
                malformed += 1
                conversation += [
                    ("assistant", [{"type": "text", "text": text}]),
                    ("user", [{"type": "text", "text": "That answer was invalid: " + str(exc) + ". Reply again with only the JSON object."}]),
                ]
        counts = [attempt["usage"] for attempt in attempts if attempt["usage"] is not None]
        response = {
            "model": self.provider + ":" + self.model,
            "answers": answers,
            "usage": {
                "input_tokens": _total(counts, "input_tokens"),
                "output_tokens": _total(counts, "output_tokens"),
                "n_retries": retries,
                "n_retries_malformed": malformed,
                "latency": time.perf_counter() - started,
            },
            "debug": {"llm_attempts": attempts, "probability_source": "llm_self_reported" if self.answer_mode == "probabilities" else "llm_discrete", "normalized": normalized},
        }
        return to_public(body, response) if public else response

    def _schema(self, keys):
        def strict(properties):
            return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

        if self.answer_mode == "discrete":
            return strict({alias: {"type": "string", "enum": names} for alias, names in keys.items()})
        return strict({alias: strict({name: {"type": "number"} for name in names}) for alias, names in keys.items()})

    def _answer(self, question, value):
        if self.answer_mode == "discrete":
            if value not in question.keys:
                raise ValueError("each answer must be one of its candidate keys")
            return from_probabilities(question, [float(key == value) for key in question.keys])
        if not isinstance(value, dict) or set(value) != set(question.keys):
            raise ValueError("each question needs one probability per candidate key")
        probs = [value[key] for key in question.keys]
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in probs):
            raise ValueError("probabilities must be finite nonnegative numbers")
        if not math.isfinite(sum(probs)) or sum(probs) <= 0:
            raise ValueError("the probabilities of a question must not all be zero")
        if not self.normalize_probabilities and (any(p > 1 for p in probs) or not math.isclose(sum(probs), 1, abs_tol=1e-5, rel_tol=0)):
            raise ValueError("probabilities must sum to 1 (normalization disabled)")
        return from_probabilities(question, [float(p) for p in probs])

    def _evidence(self, state, root):
        if isinstance(state, str):
            return [{"type": "text", "text": state}]
        messages = state["messages"]
        labelled = len(messages) > 1 or messages[0]["role"] != "user"
        parts = []
        for message in messages:
            if labelled:
                parts.append({"type": "text", "text": message["role"] + ":"})
            content = message["content"]
            for item in [{"type": "text", "text": content}] if isinstance(content, str) else content:
                if item["type"] == "text":
                    parts.append(item)
                elif item["type"] == "image_url":
                    parts.append(_image(item["image_url"]["url"], root))
                else:
                    path = Path(item["video_url"]["url"])
                    parts += _video_frames(path if path.is_absolute() else root / path, self.video_frames)
        return parts

    def _payload(self, conversation, schema):
        system = SYSTEM + (DISCRETE if self.answer_mode == "discrete" else PROBABILITIES)
        if not self.structured:
            system += " Reply with only a JSON object matching this JSON Schema:\n" + json.dumps(schema)
        if self.provider == "anthropic":
            payload = {
                "model": self.model,
                "max_tokens": self.max_tokens or 4096,
                "system": system,
                "messages": [{"role": role, "content": [_anthropic_part(p) for p in parts]} for role, parts in conversation],
            }
            if self.structured:
                payload["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        elif self.provider == "gemini":
            payload = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [
                    {"role": "model" if role == "assistant" else role, "parts": [_gemini_part(p) for p in parts]}
                    for role, parts in conversation
                ],
                "generationConfig": {},
            }
            if self.structured:
                payload["generationConfig"] = {"responseMimeType": "application/json", "responseJsonSchema": schema}
            if self.max_tokens:
                payload["generationConfig"]["maxOutputTokens"] = self.max_tokens
        else:
            payload = {
                "model": self.model,
                "messages": [{"role": "system", "content": system}] + [
                    {"role": role, "content": parts[0]["text"] if role == "assistant" else [_openai_part(p) for p in parts]}
                    for role, parts in conversation
                ],
            }
            if self.structured:
                payload["response_format"] = {
                    "type": "json_schema", "json_schema": {"name": "answers", "strict": True, "schema": schema}
                }
            if self.max_tokens:
                payload["max_completion_tokens"] = self.max_tokens
        return {**payload, **self.extra_body}

    def _call(self, conversation, schema, attempts, retries):
        """One provider answer, retrying transient failures; every request is recorded."""
        payload = self._payload(conversation, schema)
        path = {"openai": "chat/completions", "anthropic": "messages"}.get(
            self.provider, "models/" + self.model + ":generateContent"
        )
        for attempt in range(self.max_retries + 1):
            record = {"request": payload, "response": None, "status_code": None, "usage": None, "error": None}
            attempts.append(record)
            started = time.perf_counter()
            try:
                remaining = _remaining(_deadline.get())
                timeout = min(self.timeout, remaining) if remaining is not None and self.timeout is not None else (remaining or self.timeout)
                response = self.client.post(path, json=payload, timeout=timeout)
                _remaining(_deadline.get())
            except httpx.TransportError as exc:
                record.update(error=type(exc).__name__, latency=time.perf_counter() - started)
                if attempt == self.max_retries:
                    raise ValenAPIError("Unable to reach " + self.provider, code="timeout" if isinstance(exc, httpx.TimeoutException) else "network_error", retryable=True, debug={"llm_attempts": attempts}) from exc
                retries += 1
                self._sleep(_delay(attempt))
                continue
            record.update(status_code=response.status_code, latency=time.perf_counter() - started,
                          request_id=response.headers.get("x-request-id") or response.headers.get("request-id"))
            _emit(self.on_event, event="attempt", attempt=len(attempts), status_code=response.status_code,
                  request_id=record["request_id"])
            try:
                record["response"] = response.json()
            except ValueError:
                record["response"] = response.text
            if response.status_code in RETRYABLE and attempt < self.max_retries:
                record["error"] = "HTTP " + str(response.status_code)
                retries += 1
                self._sleep(_delay(attempt, response))
                continue
            if response.is_error:
                detail = record["response"].get("error", {}) if isinstance(record["response"], dict) else {}
                message = detail.get("message") if isinstance(detail, dict) else None
                record["error"] = message or "HTTP " + str(response.status_code)
                raise ValenAPIError(
                    self.provider + " request failed: " + record["error"],
                    status_code=response.status_code,
                    request_id=record["request_id"],
                    debug={"llm_attempts": attempts},
                )
            text, usage, stop = _read(self.provider, record["response"])
            record["usage"] = usage
            if stop:
                record["error"] = stop
                raise ValenAPIError(self.provider + " " + stop, debug={"llm_attempts": attempts})
            return text, retries

    def _sleep(self, delay):
        remaining = _remaining(_deadline.get())
        if remaining is not None and delay >= remaining:
            raise ValenTimeoutError(debug={"reason": "retry exceeds total budget"}, model=self.model)
        _emit(self.on_event, event="retry", delay=delay)
        time.sleep(delay)

    def close(self):
        if not self.closed:
            self.closed = True
            self.client.close()


def _read(provider, data):
    """Text, token usage and a terminal stop reason (refusal or truncation), if any."""
    if provider == "anthropic":
        usage = data.get("usage") or {}
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        reason = data.get("stop_reason")
        stop = {
            "max_tokens": "output reached max_tokens; raise max_tokens or ask fewer questions",
            "refusal": "refused to answer",
            "model_context_window_exceeded": "input exceeds the model's context window",
        }.get(reason)
        return text, {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}, stop
    if provider == "gemini":
        usage = data.get("usageMetadata") or {}
        # Gemini bills thinking tokens as output, as OpenAI and Anthropic count them.
        output = usage.get("candidatesTokenCount")
        counts = {"input_tokens": usage.get("promptTokenCount"),
                  "output_tokens": None if output is None else output + usage.get("thoughtsTokenCount", 0)}
        if not data.get("candidates"):
            return "", counts, "blocked the prompt: " + str((data.get("promptFeedback") or {}).get("blockReason"))
        candidate = data["candidates"][0]
        parts = (candidate.get("content") or {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))
        reason = candidate.get("finishReason", "STOP")
        if reason == "MAX_TOKENS":
            return text, counts, "output reached max_tokens; raise max_tokens or ask fewer questions"
        return text, counts, None if reason == "STOP" else "stopped generating: " + reason
    usage = data.get("usage") or {}
    counts = {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")}
    choice = data["choices"][0]
    message = choice.get("message") or {}
    if message.get("refusal"):
        return "", counts, "refused to answer: " + message["refusal"]
    stop = {
        "length": "output reached max_tokens; raise max_tokens or ask fewer questions",
        "content_filter": "stopped generating: content_filter",
    }.get(choice.get("finish_reason"))
    return message.get("content") or "", counts, stop


def _parse_json(text):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the reply contains no JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("the reply is not valid JSON (" + exc.msg + ")") from None
    if not isinstance(value, dict):
        raise ValueError("the reply must be a JSON object")
    return value


def _total(counts, key):
    values = [count.get(key) for count in counts]
    return None if not values or None in values else sum(values)


def _image(url, root):
    if url.startswith("data:"):
        header, data = url.split(",", 1)
        return {"type": "image", "mime": header.removeprefix("data:").removesuffix(";base64"), "data": data}
    if url.startswith(("https://", "http://")):
        return {"type": "image_url", "url": url}
    path = Path(url) if Path(url).is_absolute() else root / url
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return {"type": "image", "mime": mime, "data": base64.b64encode(path.read_bytes()).decode("ascii")}


def _video_frames(path, count, max_side=768):
    """Evenly spaced JPEG frames, labelled with their timestamps."""
    try:
        import av
    except ImportError as exc:
        raise ImportError("Video inputs require valen-sdk[video]") from exc
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 24.0
        total = stream.frames or sum(1 for _ in container.decode(stream))
    count = min(count, total)
    wanted = sorted({round(i * (total - 1) / max(count - 1, 1)) for i in range(count)})
    parts = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                image = frame.to_image()
                image.thumbnail((max_side, max_side))
                buffer = BytesIO()
                image.convert("RGB").save(buffer, format="JPEG", quality=85)
                parts.append({"type": "text", "text": "Video frame at %.1f s:" % (index / fps)})
                parts.append({"type": "image", "mime": "image/jpeg", "data": base64.b64encode(buffer.getvalue()).decode("ascii")})
            if index >= wanted[-1]:
                break
    return parts


def _openai_part(part):
    if part["type"] == "text":
        return {"type": "text", "text": part["text"]}
    url = part["url"] if part["type"] == "image_url" else "data:" + part["mime"] + ";base64," + part["data"]
    return {"type": "image_url", "image_url": {"url": url}}


def _anthropic_part(part):
    if part["type"] == "text":
        return {"type": "text", "text": part["text"]}
    if part["type"] == "image_url":
        return {"type": "image", "source": {"type": "url", "url": part["url"]}}
    return {"type": "image", "source": {"type": "base64", "media_type": part["mime"], "data": part["data"]}}


def _gemini_part(part):
    if part["type"] == "text":
        return {"text": part["text"]}
    if part["type"] == "image_url":
        raise ValueError("Gemini needs inline images; pass a data URL or a local path")
    return {"inlineData": {"mimeType": part["mime"], "data": part["data"]}}


_deadline = contextvars.ContextVar("valen_llm_deadline", default=None)
_history = contextvars.ContextVar("valen_llm_history", default=None)


class AsyncLLMBackend(LLMBackend):
    """Native async provider I/O; cancellation stops retry and correction loops."""

    def __init__(self, provider, model, **options):
        transport = options.pop("transport", None)
        super().__init__(provider, model, **options)
        headers, base_url = dict(self.client.headers), self.client.base_url
        self.client.close()
        self.client = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=self.timeout, transport=transport)

    async def predict(self, body, protocol, *, media_root="."):
        if self.closed:
            raise RuntimeError("Valen client is closed")
        total = _positive(body.get("total_timeout", self.total_timeout), "total_timeout")
        token = _deadline.set(None if total is None else time.monotonic()+total)
        history_token = _history.set([])
        started = time.monotonic()
        _emit(self.on_event, event="start", backend="llm", model=self.model)
        try:
            pending = self._predict(body, protocol, media_root=media_root)
            result = await pending if total is None else await asyncio.wait_for(pending, total)
            _emit(self.on_event, event="complete", model=result["model"], usage=result["usage"], elapsed=time.monotonic()-started)
            return result
        except asyncio.TimeoutError as exc:
            _emit(self.on_event, event="error", code="timeout", elapsed=time.monotonic()-started)
            raise ValenTimeoutError(model=self.model, debug={"llm_attempts": _history.get()}) from exc
        except asyncio.CancelledError:
            _emit(self.on_event, event="error", code="cancelled", elapsed=time.monotonic()-started)
            raise
        except Exception as exc:
            if isinstance(exc, ValenAPIError):
                exc.debug = {**(exc.debug or {}), "llm_attempts": _history.get()}
                exc.model = self.provider + ":" + self.model
            _emit(self.on_event, event="error", code=getattr(exc, "code", "request_error"), elapsed=time.monotonic()-started)
            raise
        finally:
            _deadline.reset(token)
            _history.reset(history_token)

    async def _predict(self, body, protocol, *, media_root="."):
        # Share the exact question, correction, accounting and response state machine.
        steps = self._prediction(body, protocol, media_root=media_root)
        try:
            pending = next(steps)
            while True:
                pending = steps.send(await self._call(*pending))
        except StopIteration as done:
            return done.value
        finally:
            steps.close()

    async def _call(self, conversation, schema, attempts, retries):
        """One provider answer, retrying transient failures; every request is recorded."""
        payload = self._payload(conversation, schema)
        path = {"openai": "chat/completions", "anthropic": "messages"}.get(
            self.provider, "models/" + self.model + ":generateContent"
        )
        for attempt in range(self.max_retries + 1):
            record = {"request": payload, "response": None, "status_code": None, "usage": None, "error": None}
            attempts.append(record)
            started = time.perf_counter()
            try:
                remaining = _remaining(_deadline.get())
                timeout = min(self.timeout, remaining) if remaining is not None and self.timeout is not None else (remaining or self.timeout)
                response = await self.client.post(path, json=payload, timeout=timeout)
                _remaining(_deadline.get())
            except httpx.TransportError as exc:
                record.update(error=type(exc).__name__, latency=time.perf_counter() - started)
                if attempt == self.max_retries:
                    raise ValenAPIError("Unable to reach " + self.provider, code="timeout" if isinstance(exc, httpx.TimeoutException) else "network_error", retryable=True, debug={"llm_attempts": attempts}) from exc
                retries += 1
                await self._sleep(_delay(attempt))
                continue
            record.update(status_code=response.status_code, latency=time.perf_counter() - started,
                          request_id=response.headers.get("x-request-id") or response.headers.get("request-id"))
            _emit(self.on_event, event="attempt", attempt=len(attempts), status_code=response.status_code,
                  request_id=record["request_id"])
            try:
                record["response"] = response.json()
            except ValueError:
                record["response"] = response.text
            if response.status_code in RETRYABLE and attempt < self.max_retries:
                record["error"] = "HTTP " + str(response.status_code)
                retries += 1
                await self._sleep(_delay(attempt, response))
                continue
            if response.is_error:
                detail = record["response"].get("error", {}) if isinstance(record["response"], dict) else {}
                message = detail.get("message") if isinstance(detail, dict) else None
                record["error"] = message or "HTTP " + str(response.status_code)
                raise ValenAPIError(
                    self.provider + " request failed: " + record["error"],
                    status_code=response.status_code,
                    request_id=record["request_id"],
                    debug={"llm_attempts": attempts},
                )
            text, usage, stop = _read(self.provider, record["response"])
            record["usage"] = usage
            if stop:
                record["error"] = stop
                raise ValenAPIError(self.provider + " " + stop, debug={"llm_attempts": attempts})
            return text, retries


    async def _sleep(self, delay):
        remaining = _remaining(_deadline.get())
        if remaining is not None and delay >= remaining:
            raise ValenTimeoutError(model=self.model)
        _emit(self.on_event, event="retry", delay=delay)
        await asyncio.sleep(delay)

    async def close(self):
        if not self.closed:
            self.closed = True
            await self.client.aclose()
