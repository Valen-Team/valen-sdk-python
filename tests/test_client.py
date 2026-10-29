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


def test_non_json_error_page_keeps_status_and_text(monkeypatch):
    monkeypatch.setattr("valen_sdk.client.time.sleep", lambda _: None)
    page = lambda _: httpx.Response(502, text="<html>502 Bad Gateway</html>", headers={"x-request-id": "request-2"})
    with Valen.remote("https://example.test/v1", transport=httpx.MockTransport(page)) as client:
        with pytest.raises(ValenAPIError, match="Bad Gateway") as exc:
            client.system_one("red", {})
    assert (exc.value.status_code, exc.value.request_id) == (502, "request-2")
    assert isinstance(exc.value.__cause__, ValueError)


def test_user_agent_names_the_sdk_version():
    from valen_sdk import __version__
    seen = []
    def handle(request):
        seen.append(request.headers["user-agent"])
        return httpx.Response(200, json=response())
    with Valen.remote("https://example.test/v1", transport=httpx.MockTransport(handle)) as client:
        client.decisions(input="red", questions=[])
    assert seen == ["valen-sdk-python/" + __version__]


@pytest.mark.parametrize("status,expected", [(401, 1), (503, 3)])
def test_async_retry_and_errors(status, expected, monkeypatch):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("valen_sdk.client.asyncio.sleep", no_sleep)
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(status, json={"error": "failure"}, headers={"x-request-id": "request-1"})
    async def run():
        async with AsyncValen("https://example.test/v1", transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(ValenAPIError, match="failure") as exc:
                await client.decisions(input="red", questions=[])
            assert (exc.value.status_code, exc.value.request_id) == (status, "request-1")
    asyncio.run(run())
    assert len(calls) == expected


def test_dropped_connections_are_retried_then_wrapped(monkeypatch):
    monkeypatch.setattr("valen_sdk.client.time.sleep", lambda _: None)
    async def no_sleep(_):
        pass
    monkeypatch.setattr("valen_sdk.client.asyncio.sleep", no_sleep)
    calls = []
    def drop(request):
        calls.append(request)
        raise httpx.RemoteProtocolError("Server disconnected", request=request)
    with Valen.remote("https://example.test/v1", transport=httpx.MockTransport(drop)) as client:
        with pytest.raises(ValenAPIError) as exc:
            client.system_one("red", {})
    assert isinstance(exc.value.__cause__, httpx.RemoteProtocolError)
    async def run():
        async with AsyncValen("https://example.test/v1", transport=httpx.MockTransport(drop)) as client:
            with pytest.raises(ValenAPIError):
                await client.decisions(input="red", questions=[])
    asyncio.run(run())
    assert len(calls) == 6


def test_nested_error_message_and_hosted_request_id():
    def handle(request):
        return httpx.Response(400, json={"error": {"type": "invalid_request", "message": "At most 8 questions"}},
                              headers={"x-llm-proxy-request-id": "proxy-1"})
    with Valen.remote("https://example.test/v1", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(ValenAPIError) as exc:
            client.system_one("red", {})
    assert (str(exc.value), exc.value.request_id) == ("At most 8 questions", "proxy-1")


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
