# Anvil

> **Forge anything.**
> A zero-jank, single-binary local AI runtime.

Anvil is a terminal-first local LLM engine that runs natively. One binary. Zero runtime dependencies. No background daemon. Just pure, in-process inference optimized for maximum throughput.

## Core Features

- **Zero Overhead:** No hidden blob storage, no telemetry, no always-on daemons. Runs only when you tell it to.
- **Maximum Speed:** Built-in **TurboQuant** KV cache compression and **Speculative Decoding** (MTP for Gemma 4, NextN for Qwen 3.6).
- **Hardware-Adaptive:** Auto-detects your CPU/GPU/RAM and automatically configures optimal backend and settings.
- **Seamless Ecosystem:** Pulls GGUF models directly from **Ollama** and **HuggingFace** registries with built-in resumable downloads.
- **Persistent Profiles:** Models retain their settings across runs automatically.

## Quickstart

**Install:**
```bash
curl -fsSL https://anvil-llm.github.io/anvil/install.sh | sh
```
*(On Linux x86_64, the installer automatically prefers the CUDA prebuilt with bundled runtime — requiring only the NVIDIA driver).*

**Run:**
```bash
# Run a local model
anvil run model.gguf

# Pull and run from Ollama
anvil pull ollama:llama3.2:3b
anvil run llama3.2:3b

# Interactive pull from HuggingFace
anvil pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF
```

## Advanced Usage

Anvil gives you granular control when you need it, and remembers your preferences when you don't.

```bash
# Maximize performance with TurboQuant and MTP
anvil run model.gguf --ctx 128000 --type-k turbo4 --type-v turbo3 --mtp

# Save overrides to the model's persistent profile
anvil run llama3.2 --temp 0.9 --save

# Manage profiles directly
anvil profile llama3.2 set n_ctx=131072 type_k=turbo4
```

## NVIDIA / CUDA (Linux)

Anvil requires only the NVIDIA driver (CUDA toolkit is bundled). If you are missing drivers, use our universal installer for any distro/GPU:

```bash
curl -fsSL -O https://raw.githubusercontent.com/anvil-llm/anvil/main/docs/anvil-nvidia-install.sh
sudo sh anvil-nvidia-install.sh
```

## Build from Source

```bash
git clone --recursive https://github.com/anvil-llm/anvil
cd anvil
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## License
MIT
