# Changelog

## 0.2.0 — 2026-10-26

First stable GitHub release of the modular ONNX and HTTP SDK. Promotes the 0.2.0 development API without a request/response schema change. Supports Decisions and SystemOne, local asset integrity/offline caching, and text/image/video preprocessing. Local real-model evidence is recorded separately from contract tests.

Compatibility: Python HTTP requires 3.10+; local ONNX requires 3.11+ and ONNX Runtime 1.31+. JavaScript requires Node.js 20+; the measured Node backend uses ONNX Runtime 1.30.0; browser inference uses the explicitly pinned development WebGPU runtime. Stable SDK version does not make this browser dependency stable. ONNX format_version=2, contract_version=valen-decisions-1. GGUF is served through HTTP, not Valen.local. See README for optional dependencies and backend-specific scope.

The GitHub release is independently installable. PyPI/npm publication is pending registry authorization; do not infer registry availability from a Git tag or CI build.
