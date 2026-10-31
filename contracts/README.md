# SDK behavior contract, version 1

Python and JavaScript use the byte-identical `contracts/v1.json` fixture. Change the version when changing semantics; keep the copies identical. Each repository tests request-aware response validation and probability formulas against it. Geometry and resize goldens remain in their existing tests.

## Protocol and answers

- Typed `decisions` and `system_one`/`systemOne` calls validate every requested answer: a nonempty model ID and usage object; no missing, duplicate or unexpected names; matching question kinds and candidate labels; finite, bounded probabilities; sum within absolute tolerance `1e-5` of one; bounded confidence and score. Unknown extra response fields remain compatible.
- Public Decisions answers are returned in request order, even if a service sends them in another order. This makes tuple type inference safe.
- Booleans and strings are distinct Choice values. Candidate order is meaningful. Ties choose the first candidate in request order.
- Score is the expected **index**, not the most probable level. A bimodal distribution can have a middle expected score. SDKs never automatically choose business actions or human-review thresholds.
- Invalid service distributions are rejected, never silently normalized. The LLM comparison backend separately offers configurable normalization and records it in diagnostics.
- Raw `predict` is the compatibility escape hatch; it returns backend JSON without the typed response checks. Treat this as an advanced interface.
- Missing token usage is unknown, never evidence of zero cost. Providers may report zero explicitly. LLM probabilities are self-reports, not calibrated Valen probabilities.

## HTTP controls

Python durations are **seconds**; JavaScript durations are **milliseconds**. Existing per-attempt timeout defaults remain 120 seconds; retries default to two, in addition to the first attempt. Total budgets are opt-in (`total_timeout` / `totalTimeout`) and are never sent as request fields.

HTTP 429/502/503/504 and transport failures are retried, using full-jitter exponential backoff and `Retry-After` (seconds or HTTP date). Authentication, other non-retryable status codes and malformed typed answers are not retried as network failures. LLM adapters have their documented provider-specific transient statuses and a separate malformed-output correction count.

Async Python deadlines cancel active provider I/O, retry waits and correction loops; caller cancellation propagates unchanged. JS cancellation uses `AbortSignal` and includes reading response bodies. A custom transport/fetch implementation must honor cancellation. Python synchronous deadlines cap each HTTP phase to the remaining budget, check elapsed time on return and refuse retries beyond the budget; blocking third-party transports and synchronous CPU/media work cannot be forcibly interrupted. Use async calls for an interruptible network deadline. Synchronous preparation in async LLM calls can also occupy the event loop; prepare large media before concurrent calls.

## Resources and events

- Reuse clients. Python `http_client` injection borrows the client; SDK `close` never closes it. `transport` and `http_client` are mutually exclusive. SDK-created HTTP clients are closed by the SDK.
- Close is idempotent. New calls after close fail. Finish/cancel admitted remote requests before closing; remote close does not promise to drain concurrent calls.
- Python local inference serializes calls under its model lock; close waits for that lock. JS local inference queues admitted calls and drains before closing sessions. Do not mutate request objects while calls are running.
- `warmup(request)` runs a caller-supplied representative request; it can incur a real model/provider call. There is no fabricated benchmark payload. Local `info` reports selected device/providers, variant, loaded graphs and closed state.
- `on_event` / `onEvent` receives metadata only: event names, attempt counts, retry delays, durations, request IDs, model identifiers and token usage when available. Python elapsed/delay values use seconds; JS uses milliseconds. HTTP `complete` describes transport success; typed validation can still reject that response. Callbacks must be quick. Callback failures issue a generic warning without changing outcomes.
- Local events describe resolve/download/load stages and lazy graph loading, not a guaranteed byte-progress percentage. Model asset names may appear; input text, media bytes, credentials, headers and full bodies do not.
- Explicit LLM `debug.llm_attempts` contains full provider requests/replies and may include private data. It is not emitted in default events. Store it only when appropriate for your application.

## Compatibility and releases

Keep both protocols and existing method aliases. More precise JS types are additive but may expose mistakes previously accepted by the compiler. Strict typed response validation intentionally rejects malformed responses previously passed through. Use a minor release and call out these changes before publishing.

Validate installed wheels/npm tarballs, including HTTP-only imports without local inference frameworks. Unit tests do not prove model accuracy, CUDA availability or all browser devices; real-model and device results must identify the tested artifact and platform.
