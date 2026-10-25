import asyncio
import json
import subprocess
import sys

import httpx
import pytest

from valen_sdk import AsyncValen, Choice, Noul, Score, Valen, ValenAPIError


def response(public=True):
    return {"model": "Valen", "answers": [{"name": "red", "type": "predicate", "probability": 0.8}] if public else {
        "red": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 10, "output_tokens": 0}}


def test_http_contract_and_typed_answers():
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response(request.url.path.endswith("decisions")))
    with Valen.remote("https://example.test/v1", api_key="test-key", model="Valen",
                      transport=httpx.MockTransport(handle)) as client:
        result = client.decisions(input="red", questions=[{"name": "red", "type": "predicate", "instructions": "Red?"}])
        assert result.by_name("red").probability == 0.8
        result = client.system_one("red", {"red": Noul(instructions="Red?")})
        assert result.nouls["red"].noul == 0.8
        assert requests[0].url.path == "/v1/decisions"
        assert requests[1].url.path == "/v1/systemone"
        assert requests[0].headers["authorization"] == "Bearer test-key"
        assert json.loads(requests[1].content)["questions"]["red"] == {"type": "noul", "instructions": "Red?"}
    with pytest.raises(RuntimeError, match="closed"):
        client.predict({"input": "red", "questions": []})


@pytest.mark.parametrize("status,expected", [(401, 1), (400, 1), (503, 3), (429, 3)])
def test_retry_and_errors(status, expected, monkeypatch):
    monkeypatch.setattr("valen_sdk.client.time.sleep", lambda _: None)
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "failure"}, headers={"x-request-id": "request-1"})
    with Valen.remote("https://example.test/v1", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValenAPIError) as exc:
            client.system_one("red", {"red": Noul(instructions="Red?")})
        assert exc.value.status_code == status
        assert exc.value.request_id == "request-1"
    assert len(calls) == expected


def test_async_http():
    async def run():
        async with AsyncValen("https://example.test/v1", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response()))) as client:
            result = await client.decisions(input="red", questions=[])
            assert result.by_name("red").probability == 0.8
    asyncio.run(run())


def test_import_does_not_load_inference_frameworks():
    script = "import valen_sdk,sys; assert not {'torch','transformers','onnxruntime','numpy'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", script], check=True)


def test_question_helpers():
    assert Choice(instructions="Classify", criteria={"a": "A"}).model_dump()["type"] == "choice"
    assert Score(instructions="Rate", criteria=["Low", "High"]).model_dump()["criteria"] == ["Low", "High"]
