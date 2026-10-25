# Third party components

The request compiler, protocol adapter and decision response formulas derive from the Valen project under Apache-2.0. The compiler preserves its prompt construction and span semantics while using NumPy arrays.

Qwen image patch layout, resize, video sampling/timestamps and mRoPE geometry are adapted from Hugging Face Transformers Qwen2-VL, Qwen3-VL and Qwen3.5 implementations, copyright the Qwen team, Alibaba Group and Hugging Face contributors, under Apache-2.0. The development reference is the Transformers 5.19.0 installation used for Valen validation. Runtime dependencies include Hugging Face tokenizers and Jinja2 instead of copying their engines.

The optimized graph exporter uses the Microsoft ONNX Runtime GenAI project, revision `6b7ff3ca3efa15da219050fdb4ecf8bed67875af`, under MIT. Its code and license remain in the development upstream checkout; the exporter subclass is maintained with Valen export tools.
