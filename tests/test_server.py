import asyncio
import httpx

from valen_sdk import AsyncValen, Noul, Valen
from valen_sdk.server import create_app


class Backend:
    name = "Valen-test"
    def predict(self, body, protocol):
        return {"model": self.name, "answers": [{"name": "red", "type": "predicate", "probability": 0.8}] if protocol == "decisions" else {
            "red": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 10, "output_tokens": 0}}
    def close(self):
        pass


def test_http_roundtrip_auth_models_and_media():
    async def run():
        app = create_app(Valen(Backend()), api_key="secret")
        transport = httpx.ASGITransport(app=app)
        async with AsyncValen("http://test/v1", api_key="secret", transport=transport) as client:
            result = await client.system_one("red", {"red": Noul(instructions="Red?")})
            assert result.nouls["red"].noul == 0.8
            result = await client.decisions(input="red", questions=[{"name": "red", "type": "predicate", "instructions": "Red?"}])
            assert result.by_name("red").probability == 0.8
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            assert (await client.get("/v1/models")).status_code == 401
            models = await client.get("/v1/models", headers={"authorization": "Bearer secret"})
            assert models.json()["data"][0]["id"] == "Valen-test"
            for endpoint in ["/v1/systemone", "/v1/system_one"]:
                response = await client.post(endpoint, headers={"authorization": "Bearer secret"}, json={
                    "state": "red", "questions": {"red": {"type": "noul", "instructions": "Red?"}}})
                assert response.status_code == 200
                assert response.json()["answers"]["red"]["noul"] == 0.8
            result = await client.post("/v1/decisions", headers={"authorization": "Bearer secret"}, json={
                "input": [{"role": "user", "content": [{"type": "input_image", "image_url": "/etc/passwd"}]}], "questions": []})
            assert result.status_code == 400
            for part in [None, {"type": "input_image", "image_url": None},
                         {"type": "input_video", "video_url": "/tmp/video.mp4"}]:
                result = await client.post("/v1/decisions", headers={"authorization": "Bearer secret"}, json={
                    "input": [{"role": "user", "content": [part]}], "questions": []})
                assert result.status_code == 400
    asyncio.run(run())
