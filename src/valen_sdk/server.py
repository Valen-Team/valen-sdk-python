"""Optional FastAPI service over the same local inference backend."""

from contextlib import asynccontextmanager
import secrets
import uuid


def create_app(model, *, api_key=None, allow_local_media=False, **model_options):
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse
    from .client import Valen
    from .errors import ModelCompatibilityError

    client = model if isinstance(model, Valen) else Valen.local(model, **model_options)

    @asynccontextmanager
    async def lifespan(app):
        yield
        client.close()

    app = FastAPI(title="Valen Decisions API", version="0.2.0", lifespan=lifespan)
    app.state.valen = client

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        response = await call_next(request)
        response.headers["x-request-id"] = uuid.uuid4().hex
        return response

    def authorize(authorization: str | None = Header(default=None)):
        if api_key and not secrets.compare_digest(authorization or "", "Bearer " + api_key):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def validate_media(body):
        if allow_local_media:
            return
        source = body.get("input", body.get("state"))
        messages = (
            source
            if isinstance(source, list)
            else source.get("messages", [])
            if isinstance(source, dict)
            else []
        )
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("messages must contain objects")
            content = message.get("content", [])
            for part in content if isinstance(content, list) else []:
                if not isinstance(part, dict):
                    raise ValueError("content parts must be objects")
                if part.get("type") in {"input_image", "input_video", "image_url", "video_url"}:
                    url = part.get("image_url", part.get("video_url", ""))
                    url = url.get("url", "") if isinstance(url, dict) else url
                    video = part.get("type") in {"input_video", "video_url"}
                    prefix = "data:video/" if video else "data:image/"
                    if not isinstance(url, str) or not url.startswith(prefix):
                        raise ValueError(
                            "HTTP media must use inline image/video data URLs; local paths are disabled"
                        )

    @app.exception_handler(ValueError)
    async def invalid_request(request, exc):
        return JSONResponse(
            status_code=400, content={"error": {"type": "invalid_request", "message": str(exc)}}
        )

    @app.exception_handler(ModelCompatibilityError)
    async def incompatible_model(request, exc):
        return JSONResponse(
            status_code=503, content={"error": {"type": "model_unavailable", "message": str(exc)}}
        )

    @app.exception_handler(ImportError)
    async def missing_feature(request, exc):
        return JSONResponse(
            status_code=503, content={"error": {"type": "feature_unavailable", "message": str(exc)}}
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    def models():
        name = getattr(client._backend, "name", "Valen")
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "Valen"}]}

    @app.post("/v1/decisions", dependencies=[Depends(authorize)])
    def decisions(body: dict):
        if "input" not in body:
            raise ValueError("Decisions requests require input")
        validate_media(body)
        return client._backend.predict(body, "decisions")

    @app.post("/v1/systemone", dependencies=[Depends(authorize)])
    @app.post("/v1/system_one", dependencies=[Depends(authorize)], include_in_schema=False)
    def system_one(body: dict):
        if "state" not in body:
            raise ValueError("SystemOne requests require state")
        validate_media(body)
        return client._backend.predict(body, "system_one")

    return app
