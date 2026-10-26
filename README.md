<p align="center">
  <a href="https://github.com/Valen-Team/Valen"><img src="assets/branding/valen-mark-white.png" alt="Valen" width="112"></a>
</p>
<h1 align="center">Valen Python SDK</h1>
<p align="center"><strong>Fast Decisions. Continuous Evolution.</strong></p>
<p align="center">Bring Valen’s multimodal System One decisions to Python.</p>

<p align="center">
  <strong>English</strong> · <a href="README_zh.md">简体中文</a><br>
  <a href="https://github.com/Valen-Team/Valen">Valen</a> · <a href="https://valen-team.github.io/">Try Valen</a> · <a href="https://valen-team.github.io/docs/">Documentation</a> · <a href="https://huggingface.co/Valen-Team">Models</a>
</p>

<p align="center">
  <strong>Python SDK</strong> · <a href="https://github.com/Valen-Team/valen-sdk-js">JavaScript SDK</a> · <a href="https://github.com/Valen-Team/valen-sdk-cpp">C++ integration</a>
</p>

[Valen](https://github.com/Valen-Team/Valen) is a multimodal System One decision model that returns structured decision probabilities for your tasks. This repository provides the Python SDK for Valen.

Run text, image and video decisions inside your Python application. Get categories, yes/no probabilities and scores for ticket routing, content screening and ranking. Use local CPU inference, NVIDIA CUDA or a remote HTTP service. Local inference uses ONNX Runtime without PyTorch or Transformers.

## Complete application tutorials

Complete workflows are maintained in the [official Cookbook](https://valen-team.github.io/cookbook/): [ticket routing](https://valen-team.github.io/cookbook/ticket-routing/), [product-image screening](https://valen-team.github.io/cookbook/product-screening/) and [video tagging](https://valen-team.github.io/cookbook/video-tagging/). Code, samples and measured outputs share one source. This repository retains SDK interface and installation examples. `Valen.local` loads ONNX; connect to GGUF services over HTTP. C++ support remains scoped by this repository's reference.


## Stable version 0.2.0

Published on [PyPI](https://pypi.org/project/valen-sdk/0.2.0/). Install the base package for HTTP clients, or the ONNX extra for local inference:

```sh
python -m pip install 'valen-sdk==0.2.0'
# Local CPU inference, Python 3.11+
python -m pip install 'valen-sdk[onnx]==0.2.0'
```

The [GitHub stable artifacts](https://github.com/Valen-Team/valen-sdk-python/releases/tag/v0.2.0) remain available. PyPI publication uses GitHub Trusted Publishing without a long-lived PyPI token; see [releasing](RELEASING.md). JavaScript SDK npm availability is tracked separately.

For a first try, use the [web experience or free API](https://valen-team.github.io/). See [version selection](https://valen-team.github.io/docs/valen/choosing/).

## Quick start

The repository example below needs Python 3.11+ and Git. It runs on CPU; no GPU is required. Installing the PyPI wheel itself does not require Git:

```bash
git clone --branch v0.2.0 --depth 1 https://github.com/valen-team/valen-sdk-python.git
cd valen-sdk-python
python -m pip install ".[onnx]"
python examples/local_decisions.py
```

A minimal protocol call follows; the Cookbook owns routing, thresholds and ranking:

```python
from valen_sdk import Valen

with Valen.local("Valen-Team/Valen-0.8B-ONNX", variant="fp32") as client:
    result = client.decisions(input="A cat is on the sofa.", questions=[
        {"name": "cat", "type": "predicate", "instructions": "A cat is present."},
    ])
    print(result.by_name("cat").probability)
```

The first run downloads and caches the selected variant; the full 0.8B FP32 download is about 3.51 GB. Later runs reuse the cache. See the [0.8B](https://huggingface.co/Valen-Team/Valen-0.8B-ONNX), [2B](https://huggingface.co/Valen-Team/Valen-2B-ONNX) and [4B](https://huggingface.co/Valen-Team/Valen-4B-ONNX) cards for model selection, downloads and measured memory.

To install the SDK directly in your own project:

```bash
python -m pip install "valen-sdk[onnx]==0.2.0"
```

## Images and video

Minimal local media interface examples remain in `examples/`. Complete inputs, dependencies, commands, outputs and uncertainty handling live in [product screening](https://valen-team.github.io/cookbook/product-screening/), [video tagging](https://valen-team.github.io/cookbook/video-tagging/) and [direct-video basics](https://valen-team.github.io/cookbook/local-inference/). Shared `Valen/assets/cookbook/` URLs remain stable.

## Local HTTP service

Start a CPU service on your machine:

```bash
python -m pip install ".[onnx,server]"
valen serve --model Valen-Team/Valen-0.8B-ONNX --model-name valen --port 8000
```

In another terminal, run `python examples/http_client.py`, or call the service from your application:

```python
from valen_sdk import Valen

with Valen.remote("http://127.0.0.1:8000/v1", model="valen") as client:
    result = client.decisions(
        input="Please send my order to my new address.",
        questions=[{"name": "change_address", "type": "predicate",
                    "instructions": "Is the customer requesting a delivery address change?"}],
    )
    print("Address change probability:", result.by_name("change_address").probability)
```

For an existing Valen service, install just the base SDK; the HTTP client does not need ONNX Runtime. The `Valen.remote` URL ends in `/v1`; pass `api_key` when the server uses authentication. `AsyncValen` provides an asynchronous HTTP client.

## SystemOne

Decisions uses a question list; SystemOne uses questions keyed by ID. Both formats run the same model. Run `python examples/system_one.py` independently:

```python
from valen_sdk import Choice, Noul, Score, Valen

with Valen.local("Valen-Team/Valen-0.8B-ONNX", variant="fp32") as client:
    result = client.system_one(
        "I moved. Please ship my order to 12 Elm Street instead.",
        {
            "intent": Choice(instructions="What does the customer want?", criteria={
                "address": "Change the delivery address", "cancel": "Cancel the order"}),
            "change_address": Noul(instructions="The customer requests a delivery address change."),
            "explicitness": Score(instructions="How explicit is the request?", criteria=[
                "No request", "Asks whether a change is possible", "Explicitly requests a change"]),
        },
    )
    print("Intent:", result.answers["intent"].choice)
    print(f"Address change probability: {result.answers['change_address'].noul:.3f}")
    print(f"Explicitness score (0–2): {result.answers['explicitness'].score:.3f}")
```

## CUDA

In a separate environment, install `python -m pip install ".[cuda]"`, then select the CUDA variant:

```python
from valen_sdk import Valen

with Valen.local(
    "Valen-Team/Valen-0.8B-ONNX", variant="cuda-fp16",
    providers=[("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"],
) as client:
    result = client.decisions(
        input="Please send my order to my new address.",
        questions=[{"name": "change_address", "type": "predicate",
                    "instructions": "Is the customer requesting a delivery address change?"}],
    )
    print(result.by_name("change_address").probability)
```

## Cache and options

After downloading the model, run from the cache offline:

```python
from valen_sdk import Valen

with Valen.local("Valen-Team/Valen-0.8B-ONNX", local_files_only=True) as client:
    result = client.decisions(
        input="Please send my order to my new address.",
        questions=[{"name": "change_address", "type": "predicate",
                    "instructions": "Is the customer requesting a delivery address change?"}],
    )
    print(result.by_name("change_address").probability)
```

You can also pass a local directory, such as `Valen.local("./models/Valen-0.8B-ONNX")`.

| Option | Use |
| --- | --- |
| `variant` | CPU default `fp32`; `q8-mlp` for lower memory; `cuda-fp16` for CUDA |
| `cache_dir` / `revision` | Choose a cache location or HF model revision |
| `num_threads` | CPU inference threads, default 4 |
| `execution` | Default `shared_state` shares one forward pass within a request; `question` compiles each question separately |
| `temperature` | Request temperature, default 1 |
| `media_kwargs` | Set image size, video frame count or FPS |

One local client executes requests serially. `predict` accepts raw requests and returns dictionaries; `decisions` and `system_one` return typed results. HTTP-only clients support Python 3.10+.

## More

[JavaScript / browser SDK](https://github.com/valen-team/valen-sdk-js) · [C++ text integration](https://github.com/valen-team/valen-sdk-cpp) · [Free API documentation](https://valen-team.github.io/docs/valen/api/)

Apache-2.0. See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).

User guides are maintained only on the website: [Documentation](https://valen-team.github.io/docs/) · [Fine-tuning](https://valen-team.github.io/docs/valen/training/) · [Probabilities](https://valen-team.github.io/docs/valen/probabilities/) · [Cookbook](https://valen-team.github.io/cookbook/).
