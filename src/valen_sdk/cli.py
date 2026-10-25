"""JSON inference and optional local HTTP service."""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(prog="valen")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser("predict", help="Read one JSON request from a file or stdin")
    infer.add_argument("--model")
    infer.add_argument("--base-url")
    infer.add_argument("--request", default="-")
    infer.add_argument("--revision")
    infer.add_argument("--variant", default="fp32")
    infer.add_argument("--offline", action="store_true")
    infer.add_argument("--execution", choices=["shared_state", "question"])
    infer.add_argument("--threads", type=int, default=4)
    serve = commands.add_parser("serve", help="Serve local ONNX models with FastAPI")
    serve.add_argument("--model", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--revision")
    serve.add_argument("--variant", default="fp32")
    serve.add_argument("--offline", action="store_true")
    serve.add_argument("--threads", type=int, default=4)
    serve.add_argument("--model-name")
    args = parser.parse_args()
    if args.command == "serve":
        import uvicorn
        from .server import create_app

        app = create_app(
            args.model,
            api_key=os.environ.get("VALEN_API_KEY"),
            revision=args.revision,
            variant=args.variant,
            local_files_only=args.offline,
            num_threads=args.threads,
            model_name=args.model_name,
        )
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        from .client import Valen

        if bool(args.model) == bool(args.base_url):
            parser.error("predict requires exactly one of --model or --base-url")
        client = (
            Valen.remote(args.base_url, api_key=os.environ.get("VALEN_API_KEY"))
            if args.base_url
            else Valen.local(
                args.model,
                revision=args.revision,
                variant=args.variant,
                local_files_only=args.offline,
                num_threads=args.threads,
                execution=args.execution,
            )
        )
        with client:
            if args.request == "-":
                body = json.load(sys.stdin)
            else:
                with open(args.request, encoding="utf-8") as stream:
                    body = json.load(stream)
            print(json.dumps(client.predict(body), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
