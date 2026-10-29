<p align="center">
  <a href="https://github.com/Valen-Team/Valen"><img src="assets/branding/valen-mark-white.png" alt="Valen" width="112"></a>
</p>
<h1 align="center">Valen Python SDK</h1>
<p align="center"><strong>快速决策，持续进化。</strong></p>
<p align="center">将 Valen 的多模态 System One 决策接入 Python。</p>

<p align="center">
  <a href="README.md">English</a> · <strong>简体中文</strong><br>
  <a href="https://github.com/Valen-Team/Valen">Valen 主仓库</a> · <a href="https://valen-team.github.io/zh-Hans/">在线体验</a> · <a href="https://valen-team.github.io/zh-Hans/docs/">文档</a> · <a href="https://huggingface.co/Valen-Team">模型</a>
</p>

<p align="center">
  <strong>Python SDK</strong> · <a href="https://github.com/Valen-Team/valen-sdk-js">JavaScript SDK</a> · <a href="https://github.com/Valen-Team/valen-sdk-cpp">C++ 集成</a>
</p>

[Valen（万澜）](https://github.com/Valen-Team/Valen) 是多模态 System One 决策模型，面向给定任务输出结构化决策概率。本仓库提供 Valen 的 Python SDK。

在自己的 Python 应用中处理文本、图片和视频，获取可直接用于工单路由、内容筛选与排序的类别、是非概率和评分。支持 CPU 本地推理、NVIDIA CUDA 和远程 HTTP 调用；本地推理使用 ONNX Runtime，无需安装 PyTorch 或 Transformers。

## 完整应用教程

完整业务流程只在[官网 Cookbook](https://valen-team.github.io/zh-Hans/cookbook/)维护：[工单分流](https://valen-team.github.io/zh-Hans/cookbook/ticket-routing/)、[候选项都不对](https://valen-team.github.io/zh-Hans/cookbook/unknown-categories/)、[商品图片筛选](https://valen-team.github.io/zh-Hans/cookbook/product-screening/)、[短视频标注](https://valen-team.github.io/zh-Hans/cookbook/video-tagging/)。代码、样例和实测结果来自同一维护源。本仓库保留 SDK 接口与安装示例；`Valen.local` 是 ONNX 路径，GGUF 服务使用远程 HTTP 客户端。C++ 集成的支持范围以本仓库说明为准。


## 稳定版本 0.2.0

已正式发布到 [PyPI](https://pypi.org/project/valen-sdk/0.2.0/)。仅调用 HTTP 服务可安装基础包；本地 ONNX 推理安装对应 extra：

```sh
python -m pip install 'valen-sdk==0.2.0'
# 本地 CPU 推理，Python 3.11+
python -m pip install 'valen-sdk[onnx]==0.2.0'
```

[GitHub 稳定包](https://github.com/Valen-Team/valen-sdk-python/releases/tag/v0.2.0)继续保留。PyPI 通过 GitHub Trusted Publishing 发布，无需长期 PyPI token；见[发布说明](RELEASING.md)。JavaScript SDK 的 npm 发布状态单独管理。

首次体验优先使用[网页或免费 API](https://valen-team.github.io/)。完整[版本选择](https://valen-team.github.io/zh-Hans/docs/valen/choosing/)。

## 快速开始

以下仓库示例需要 Python 3.11+ 和 Git，使用 CPU，无需 GPU；直接安装 PyPI wheel 本身不需要 Git：

```bash
git clone --branch v0.2.0 --depth 1 https://github.com/valen-team/valen-sdk-python.git
cd valen-sdk-python
python -m pip install ".[onnx]"
python examples/local_decisions.py
```

最短协议调用如下，完整分流、阈值和排序见 Cookbook：

```python
from valen_sdk import Valen

with Valen.local("Valen-Team/Valen-0.8B-ONNX", variant="fp32") as client:
    result = client.decisions(input="A cat is on the sofa.", questions=[
        {"name": "cat", "type": "predicate", "instructions": "A cat is present."},
    ])
    print(result.by_name("cat").probability)
```

首次使用会下载所选模型变体并缓存；0.8B FP32 完整下载约 3.51 GB。后续运行复用缓存。模型选择、下载量和实测内存见 [0.8B](https://huggingface.co/Valen-Team/Valen-0.8B-ONNX)、[2B](https://huggingface.co/Valen-Team/Valen-2B-ONNX)、[4B](https://huggingface.co/Valen-Team/Valen-4B-ONNX) 模型卡。

在自己的项目中，可直接安装 SDK：

```bash
python -m pip install "valen-sdk[onnx]==0.2.0"
```

## 图片和视频

本地媒体接口的最短示例保留在 `examples/`；完整输入、依赖、运行命令、结果与不确定性处理统一见[图片筛选](https://valen-team.github.io/zh-Hans/cookbook/product-screening/)、[短视频标注](https://valen-team.github.io/zh-Hans/cookbook/video-tagging/)和[直接视频基础](https://valen-team.github.io/zh-Hans/cookbook/local-inference/)。共享素材 `Valen/assets/cookbook/` 的 URL 保持不变。

## 本地 HTTP 服务

在自己的机器上启动 CPU 服务：

```bash
python -m pip install ".[onnx,server]"
valen serve --model Valen-Team/Valen-0.8B-ONNX --model-name valen --port 8000
```

另开一个终端运行 `python examples/http_client.py`，或在应用中调用：

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

已有 Valen 服务时，仅安装 SDK 基础包即可使用 HTTP 客户端，无需安装 ONNX Runtime。`Valen.remote` 的地址以 `/v1` 结尾；服务端启用鉴权时传入 `api_key`。`AsyncValen` 提供异步 HTTP 客户端。

## SystemOne

Decisions 使用问题列表；SystemOne 使用以问题 ID 为键的字典。两种格式调用同一模型，脚本 `python examples/system_one.py` 可独立运行：

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

## 与大模型对比

`Valen.llm` 把同一个 `system_one` 或 `decisions` 请求交给通用大模型回答，返回同样的类型化结果，因此路由规则、阈值和测试无需改动即可在两者之间切换。可以用它在自己的数据上衡量换成 Valen 能节省多少。该功能包含在基础包中；视频输入需安装 `valen-sdk[video]`。运行 `python examples/llm_compare.py`：

```python
from valen_sdk import Choice, Valen

questions = {"queue": Choice(instructions="Which team should handle this ticket?", criteria={
    "billing": "Payments, charges and refunds", "technical": "Bugs and crashes in the app"})}
valen = Valen.remote("https://api.valen-team.workers.dev/v1", model="valen-2b")
llm = Valen.llm("openai", "gpt-4o-mini")  # 读取 OPENAI_API_KEY
for client in (valen, llm):
    result = client.system_one("I was charged twice.", questions)
    print(result.model, result.choices["queue"].choice, result.usage.input_tokens)
```

`provider` 可选 `openai`（Chat Completions；通过 `base_url` 也可接任何兼容 OpenAI 的服务）、`anthropic` 或 `gemini`；密钥读取 `OPENAI_API_KEY`、`ANTHROPIC_API_KEY` 或 `GEMINI_API_KEY`，也可传 `api_key=`。问题 ID 不会发送给大模型。图片以内联方式发送，视频按均匀抽取的帧发送。

| 参数 | 用途 |
| --- | --- |
| `structured` | 默认 `True` 使用服务商的 JSON Schema 输出；`False` 在提示词中要求 JSON，适用于不支持结构化输出的服务 |
| `answer_mode` | `probabilities`（默认）：大模型给出每个候选的概率；`discrete`：只选一个候选，以概率 1 返回 |
| `n_retry_malformed` | 输出不符合 schema 时的纠正重试次数，默认 2 |
| `max_retries` | 限流、过载和连接失败的重试次数，默认 2 |
| `max_tokens` | 输出长度上限；Anthropic 默认 4096 |
| `video_frames` | 每个视频发送的帧数，默认 8 |
| `extra_body` | 额外的服务商参数，例如 `{"temperature": 0}` 或 `{"reasoning_effort": "low"}` |

大模型的概率是它自己报出的估计值，并非 Valen 的候选打分；请在有标注的样本上比较决策和准确率，而不是比较概率数值。`usage` 额外包含 `n_retries`、`n_retries_malformed` 和 `latency`，token 数为所有尝试之和（服务未返回时为 `None`）。`debug["llm_attempts"]` 按顺序记录每次请求和响应，把其中的 `request` 发到同一地址即可重放。输出被截断、拒答和上下文超长会直接抛出 `ValenAPIError`，不做纠正重试，尝试记录在 `error.debug` 中。

## CUDA

在独立环境中安装 `python -m pip install ".[cuda]"`，然后使用 CUDA 变体：

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

## 缓存与调用参数

模型下载后，可从缓存离线运行：

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

模型参数也可传入本地目录，如 `Valen.local("./models/Valen-0.8B-ONNX")`。

| 参数 | 用途 |
| --- | --- |
| `variant` | CPU 默认 `fp32`；节省内存选 `q8-mlp`；CUDA 选 `cuda-fp16` |
| `cache_dir` / `revision` | 指定缓存位置或 HF 模型版本 |
| `num_threads` | CPU 推理线程数，默认 4 |
| `execution` | 默认 `shared_state`，同一请求中的问题共享一次前向；`question` 为每个问题单独编译 |
| `temperature` | 请求温度，默认 1 |
| `media_kwargs` | 调整图片尺寸、视频帧数或 FPS |

本地单个客户端按序执行请求。`predict` 接受原始请求并返回字典；`decisions` 和 `system_one` 返回可按字段读取的类型化结果。仅使用 HTTP 客户端时支持 Python 3.10+。

## 更多

[JavaScript / 浏览器 SDK](https://github.com/valen-team/valen-sdk-js) · [C++ 文本集成示例](https://github.com/valen-team/valen-sdk-cpp) · [免费 API 文档](https://valen-team.github.io/docs/valen/api/) · [Agent skill](https://github.com/Valen-Team/skills) · [llms.txt](https://valen-team.github.io/zh-Hans/llms.txt)

Apache-2.0。见 [LICENSE](LICENSE) 和 [第三方许可](THIRD_PARTY_NOTICES.md)。

用户指南唯一维护在官网：[Documentation](https://valen-team.github.io/zh-Hans/docs/) · [Fine-tuning](https://valen-team.github.io/zh-Hans/docs/valen/training/) · [Probabilities](https://valen-team.github.io/zh-Hans/docs/valen/probabilities/) · [Cookbook](https://valen-team.github.io/zh-Hans/cookbook/).
