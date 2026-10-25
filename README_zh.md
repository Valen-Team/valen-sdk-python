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

## 快速开始

本地推理需要 Python 3.11+ 和 Git。下面使用 CPU，无需 GPU：

```bash
git clone https://github.com/valen-team/valen-sdk-python.git
cd valen-sdk-python
python -m pip install ".[onnx]"
python examples/local_decisions.py
```

首个示例的完整代码：

```python
from valen_sdk import Valen

with Valen.local("Valen-Team/Valen-0.8B-ONNX", variant="fp32") as client:
    result = client.decisions(
        input="I moved. Please ship my order to 12 Elm Street instead.",
        questions=[
            {"name": "intent", "type": "choice", "instructions": "What does the customer want?",
             "choices": [{"value": "address", "description": "Change the delivery address"},
                         {"value": "cancel", "description": "Cancel the order"}]},
            {"name": "change_address", "type": "predicate",
             "instructions": "The customer requests a delivery address change."},
            {"name": "explicitness", "type": "score", "instructions": "How explicit is the request?",
             "levels": [{"label": "none", "description": "No request"},
                        {"label": "possible", "description": "Asks whether a change is possible"},
                        {"label": "explicit", "description": "Explicitly requests a change"}]},
        ],
    )
    print("Intent:", result.by_name("intent").choice)
    print(f"Address change probability: {result.by_name('change_address').probability:.3f}")
    print(f"Explicitness score (0–2): {result.by_name('explicitness').score:.3f}")
```

0.8B FP32 示例输出：

```text
Intent: address
Address change probability: 0.998
Explicitness score (0–2): 1.921
```

类别用于选择处理分支；是非概率范围为 0–1；示例评分范围为 0–2。一个输入可以同时提出多个问题。具体数值会随运行精度变化。

首次使用会下载所选模型变体并缓存；0.8B FP32 完整下载约 3.51 GB。后续运行复用缓存。模型选择、下载量和实测内存见 [0.8B](https://huggingface.co/Valen-Team/Valen-0.8B-ONNX)、[2B](https://huggingface.co/Valen-Team/Valen-2B-ONNX)、[4B](https://huggingface.co/Valen-Team/Valen-4B-ONNX) 模型卡。

在自己的项目中，可直接安装 SDK：

```bash
python -m pip install "valen-sdk[onnx] @ git+https://github.com/valen-team/valen-sdk-python.git"
```

## 图片和视频

以下命令在本仓库根目录运行。每个脚本自行创建并关闭客户端，可以独立执行。

图片：

```bash
curl -L https://raw.githubusercontent.com/Valen-Team/Valen/main/assets/cookbook/sheep.jpg -o sheep.jpg
python examples/media.py sheep.jpg
```

视频需额外安装解码依赖；示例采样四帧：

```bash
python -m pip install ".[onnx,video]"
curl -L https://raw.githubusercontent.com/Valen-Team/Valen/main/assets/cookbook/momos.mp4 -o momos.mp4
python examples/media.py momos.mp4 --video
```

查看 [媒体示例源码](examples/media.py) 了解 `input_image` / `input_video` 消息格式。本地路径相对于运行目录解析。

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

[JavaScript / 浏览器 SDK](https://github.com/valen-team/valen-sdk-js) · [C++ 文本集成示例](https://github.com/valen-team/valen-sdk-cpp) · [免费 API 文档](https://valen-team.github.io/docs/valen/api/)

Apache-2.0。见 [LICENSE](LICENSE) 和 [第三方许可](THIRD_PARTY_NOTICES.md)。
