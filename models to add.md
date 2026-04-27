# Models to Add

## Text / LLM

### OpenAI GPT-OSS 20B
- Hugging Face: `openai/gpt-oss-20b`
- License: Apache 2.0
- Shape: 21B total parameters, about 3.6B active parameters per token, 128k context.
- Hardware note: OpenAI says the native MXFP4 release only requires about 16 GB of memory, so this is the practical GPT-OSS candidate for local testing and mid/high VRAM cards.
- UI note: expose reasoning effort as a control (`low`, `medium`, `high`) rather than a plain thinking toggle once the runner supports the harmony format.

### OpenAI GPT-OSS 120B
- Hugging Face: `openai/gpt-oss-120b`
- License: Apache 2.0
- Shape: 117B total parameters, about 5.1B active parameters per token, 128k context.
- Hardware note: OpenAI says this fits in a single 80 GB GPU with MXFP4, making it a high-end option for H100/H200 class systems and larger-memory workstation/server GPUs.
- Runner note: treat as a future vLLM / harmony-format lane rather than the default Transformers runner.

Sources:
- https://openai.com/index/introducing-gpt-oss
- https://huggingface.co/openai/gpt-oss-20b
- https://huggingface.co/openai/gpt-oss-120b
