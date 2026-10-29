import base64
import json

import httpx
import pytest

from valen_sdk import Choice, Noul, Score, Valen, ValenAPIError

ANSWERS = {"q1": {"billing": 0.6, "technical": 0.3, "other": 0.1}, "q2": {"true": 0.8, "false": 0.2},
           "q3": {"0": 0.1, "1": 0.2, "2": 0.7}}
QUESTIONS = {
    "category": Choice(instructions="What is this ticket about?",
                       criteria={"billing": "Billing", "technical": "Technical", "other": "Other"}),
    "urgent": Noul(instructions="Is the customer asking for a fast fix?"),
    "frustration": Score(instructions="How frustrated is the customer?", criteria=["Calm", "Annoyed", "Angry"]),
}
PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG").decode()


def reply(provider, text, finish=None):
    if provider == "anthropic":
        return {"content": [{"type": "text", "text": text}], "stop_reason": finish or "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 20}}
    if provider == "gemini":
        return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish or "STOP"}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 15, "thoughtsTokenCount": 5}}
    return {"choices": [{"message": {"content": text}, "finish_reason": finish or "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def client(provider, handler, **options):
    return Valen.llm(provider, "test-model", api_key="test-key", transport=httpx.MockTransport(handler), **options)


@pytest.mark.parametrize("provider,path,auth", [
    ("openai", "/v1/chat/completions", ("authorization", "Bearer test-key")),
    ("anthropic", "/v1/messages", ("x-api-key", "test-key")),
    ("gemini", "/v1beta/models/test-model:generateContent", ("x-goog-api-key", "test-key")),
])
def test_system_one_matches_valen_response(provider, path, auth):
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=reply(provider, json.dumps(ANSWERS)))
    with client(provider, handle) as valen:
        state = {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "I was charged twice."}, {"type": "image_url", "image_url": {"url": PNG}}]}]}
        result = valen.system_one(state, QUESTIONS)
    assert result.model == provider + ":test-model"
    assert result.choices["category"].choice == "billing"
    assert result.choices["category"].confidence == pytest.approx((0.6 - 1 / 3) / (2 / 3))
    assert result.nouls["urgent"].noul == pytest.approx(0.8)
    assert result.answers["frustration"].score == pytest.approx(1.6)
    assert result.usage.input_tokens == 100 and result.usage.output_tokens == 20 and result.usage.n_retries == 0
    assert requests[0].url.path == path
    assert requests[0].headers[auth[0]] == auth[1]
    sent = json.dumps(json.loads(requests[0].content))
    assert "category" not in sent and "I was charged twice." in sent and "iVBORw" in sent
    assert result.debug["llm_attempts"][0]["request"] == json.loads(requests[0].content)


def test_decisions_protocol_and_discrete_mode():
    def handle(request):
        return httpx.Response(200, json=reply("openai", '```json\n{"q1": "true", "q2": "blue"}\n```'))
    with client("openai", handle, answer_mode="discrete", structured=False) as valen:
        result = valen.decisions(input="A red car", questions=[
            {"name": "red", "type": "predicate", "instructions": "Is the car red?"},
            {"name": "color", "type": "choice", "instructions": "Which colour?", "choices": [
                {"value": "red", "description": "Red"}, {"value": "blue", "description": "Blue"}]},
        ])
    assert result.by_name("red").probability == 1.0
    assert result.by_name("color").choice == "blue"


def test_malformed_output_is_corrected_then_counted():
    bodies = []
    replies = iter(["not json", json.dumps({"q1": {"true": 0, "false": 0}}), json.dumps({"q1": {"true": 3, "false": 1}})])
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=reply("openai", next(replies)))
    with client("openai", handle) as valen:
        result = valen.system_one("red", {"red": Noul(instructions="Red?")})
    assert result.nouls["red"].noul == pytest.approx(0.75)
    assert result.usage.n_retries_malformed == 2 and result.usage.input_tokens == 300
    assert bodies[2]["messages"][-1]["content"][0]["text"].startswith("That answer was invalid")
    assert bodies[2]["messages"][-2] == {"role": "assistant", "content": json.dumps({"q1": {"true": 0, "false": 0}})}


def test_malformed_output_gives_up():
    with client("openai", lambda _: httpx.Response(200, json=reply("openai", "{}")), n_retry_malformed=1) as valen:
        with pytest.raises(ValenAPIError, match="answer schema") as exc:
            valen.system_one("red", {"red": Noul(instructions="Red?")})
    assert len(exc.value.debug["llm_attempts"]) == 2


@pytest.mark.parametrize("provider,finish,message", [
    ("anthropic", "max_tokens", "max_tokens"), ("anthropic", "refusal", "refused"),
    ("gemini", "MAX_TOKENS", "max_tokens"), ("gemini", "SAFETY", "SAFETY"), ("openai", "length", "max_tokens"),
])
def test_truncation_and_refusal_are_errors_without_retry(provider, finish, message):
    calls = []
    def handle(request):
        calls.append(request)
        return httpx.Response(200, json=reply(provider, "{", finish))
    with client(provider, handle) as valen:
        with pytest.raises(ValenAPIError, match=message) as exc:
            valen.system_one("red", {"red": Noul(instructions="Red?")})
    assert len(calls) == 1
    assert exc.value.debug["llm_attempts"][0]["response"] == reply(provider, "{", finish)


def test_transient_failures_retry(monkeypatch):
    monkeypatch.setattr("valen_sdk.llm.time.sleep", lambda _: None)
    statuses = iter([529, 503, 200])
    def handle(request):
        status = next(statuses)
        return httpx.Response(status, json=reply("anthropic", json.dumps({"q1": {"true": 1, "false": 0}})) if status == 200
                              else {"error": {"message": "overloaded"}})
    with client("anthropic", handle) as valen:
        result = valen.system_one("red", {"red": Noul(instructions="Red?")})
    assert result.usage.n_retries == 2 and len(result.debug["llm_attempts"]) == 3


def test_provider_error_message_is_kept():
    def handle(request):
        return httpx.Response(401, json={"error": {"message": "Incorrect API key"}})
    with client("openai", handle) as valen:
        with pytest.raises(ValenAPIError, match="Incorrect API key") as exc:
            valen.system_one("red", {"red": Noul(instructions="Red?")})
    assert exc.value.status_code == 401


def test_unreported_usage_and_extra_body():
    bodies = []
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({"q1": {"true": 1, "false": 1}})}}]})
    with client("openai", handle, base_url="https://llm.example.test/v1", extra_body={"temperature": 0}) as valen:
        result = valen.system_one("red", {"red": Noul(instructions="Red?")})
    assert result.usage.input_tokens is None
    assert bodies[0]["temperature"] == 0


def test_video_frames(tmp_path):
    av = pytest.importorskip("av")
    pytest.importorskip("PIL")
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for i in range(12):
            frame = av.VideoFrame.from_ndarray(__import__("numpy").full((64, 64, 3), i * 20, "uint8"), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    bodies = []
    def handle(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=reply("openai", json.dumps({"q1": {"true": 1, "false": 0}})))
    with client("openai", handle, video_frames=3) as valen:
        valen.decisions(input=[{"role": "user", "content": [{"type": "input_video", "video_url": str(path)}]}],
                        questions=[{"name": "dark", "type": "predicate", "instructions": "Is the clip dark?"}])
    content = bodies[0]["messages"][1]["content"]
    assert [part["type"] for part in content].count("image_url") == 3
    assert content[0]["text"] == "Video frame at 0.0 s:"
