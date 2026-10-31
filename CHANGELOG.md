# Unreleased

- Validate typed answers against request names, kinds, candidates and finite probability distributions. Raw `predict` remains available. Malformed responses previously accepted now fail explicitly.
- Add total request budgets, Retry-After with jitter, stable timeout/response errors and metadata-only event hooks.
- Add local loading stages, runtime information and representative-request warmup; document concurrency and ownership.
- Add versioned cross-language protocol/formula fixtures and installed-package smoke tests.
- Add native asynchronous LLM comparisons, optional probability normalization and provenance diagnostics. Missing token counts are unknown, not zero.
- Support borrowed httpx clients without transferring ownership; improve missing-answer lookup errors.

# Changelog

## 0.3.1 — 2026-10-29

- `AsyncValen` retries rate limits, overload and connection failures like `Valen.remote`, takes `max_retries` and `headers`, and raises `ValenAPIError` with the service's error detail.
- A non-JSON response, such as a gateway's HTML 502, raises `ValenAPIError` with its status code, request ID and text.
- A connection dropped mid-response (`ReadError`, `RemoteProtocolError`) is retried like a connection failure and then raises `ValenAPIError` instead of an httpx exception.
- A self-hosted server's `{"error": {"message": ...}}` becomes the error message, and the hosted APIs' `x-llm-proxy-request-id` fills `request_id`.
- HTTP clients send `User-Agent: valen-sdk-python/<version>`.

## 0.3.0 — 2026-10-29

Adds `Valen.llm(provider, model)`: the same Decisions and SystemOne requests answered by an OpenAI-compatible, Anthropic or Gemini model, returning the same typed responses for side-by-side comparison. Records token usage across retries, latency and every provider attempt in `debug["llm_attempts"]`; corrects malformed output; reports truncation and refusals as `ValenAPIError` with `error.debug`. Video frames are sent as images. `valen-sdk[video]` now includes Pillow. `Usage` token counts may be `None` when a provider omits them. No new base dependencies.

Published to PyPI as `valen-sdk==0.3.0` on 2026-10-29 through GitHub Trusted Publishing from tag `v0.3.0`. The 43.7 KB wheel passed SHA256 verification, matched the tagged source, and passed the contract tests installed in a clean environment. See `validation/pypi-0.3.0.json`.

## 0.2.0 — 2026-10-26

First stable GitHub release of the modular ONNX and HTTP SDK. Promotes the 0.2.0 development API without a request/response schema change. Supports Decisions and SystemOne, local asset integrity/offline caching, and text/image/video preprocessing. Local real-model evidence is recorded separately from contract tests.

Compatibility: Python HTTP requires 3.10+; local ONNX requires 3.11+ and ONNX Runtime 1.31+. JavaScript requires Node.js 20+; the measured Node backend uses ONNX Runtime 1.30.0; browser inference uses the explicitly pinned development WebGPU runtime. Stable SDK version does not make this browser dependency stable. ONNX format_version=2, contract_version=valen-decisions-1. GGUF is served through HTTP, not Valen.local. See README for optional dependencies and backend-specific scope.

Published to PyPI as `valen-sdk==0.2.0` on 2026-10-26 through GitHub Trusted Publishing. The wheel and sdist registry metadata were checked; the downloaded 36.7 KB wheel passed SHA256 verification, matched the GitHub release Python source, installed offline without dependency downloads, and passed all 28 contract tests. See `validation/pypi-0.2.0.json`. JavaScript npm publication remains separate.
