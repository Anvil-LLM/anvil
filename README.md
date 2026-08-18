# Anvil

> **Forge anything.**
> A zero-jank, single-binary local AI runtime.

![Anvil Hero Image](https://via.placeholder.com/1200x300.png?text=Anvil+-+Forge+Anything)

## Overview

Anvil is a terminal-first local LLM engine engineered for uncompromising performance, developer control, and user simplicity. Built as a single native C++ binary, Anvil completely rethinks the local inference stack. We refuse to accept slow defaults, opaque client-server architectures, hidden background daemons, or telemetry.

Instead, Anvil delivers pure, raw, in-process inference. Weighing in at roughly 10 MB, it runs only when you tell it to, immediately unlocking the maximum potential of your local hardware.

Whether you are a researcher needing granular control over KV-cache quantization, or a user who just wants to chat with the latest Llama model without configuration headaches, Anvil is built for you.

---

## Core Positioning & Philosophy

Anvil is built on a few unshakeable principles:

1.  **Single Native Binary:** Distribution is effortless. You download one executable (~10 MB) that contains everything you need. No python environments, no container runtimes, no missing dynamically linked libraries.
2.  **Zero Overhead Runtime:** Anvil respects your machine. There is no hidden blob storage hoarding your SSD space. There is no telemetry phoning home. There is no mandatory always-on background daemon consuming RAM. When you close Anvil, it is completely gone.
3.  **In-Process Inference:** By operating entirely locally and natively, Anvil avoids the latency and complexity of unnecessary network loops and HTTP/JSON abstraction layers.
4.  **Hardware-Adaptive, Not Magic:** Anvil respects the developer. It detects your CPU, GPU, and RAM to *suggest* appropriate backend settings, but it never relies on silent "magic defaults" that guess your intent and hide the details.

---

## The Anvil Configuration Workflow

Anvil achieves the delicate balance of being both incredibly accessible and deeply configurable by strictly separating initial setup from persistent configuration and per-run overrides.

We **do not** use opaque heuristics to guess your desired setup. Instead, we use a transparent, explicit workflow:

1.  **Interactive Launch:** When you run a model for the first time, Anvil presents a beautiful, interactive TUI (Terminal User Interface).
2.  **Explicit Choice:** In the TUI, you explicitly select your hardware backend, memory allocation, and quantization preferences.
3.  **Persistent Profiles:** Anvil saves your explicit choices into a transparent JSON profile (`~/.anvil/models.json`).
4.  **Frictionless Re-runs:** On subsequent runs, Anvil seamlessly loads the saved JSON profile, booting instantly.
5.  **Granular CLI Overrides:** Developers can always bypass or modify the persistent profile by passing specific CLI flags on a per-run basis.

**Summary of Control:**
*   **TUI:** Your entry point for easy initial configuration and profile management.
*   **JSON Profiles:** Your persistent, human-readable configuration state.
*   **CLI Flags:** Your mechanism for absolute, immediate control and experimentation.

---

## Unrivaled Speed & Capabilities

Anvil doesn't just run models; it pushes the bleeding edge of local inference optimization. Our integrated `llama-turbo` backend powers advanced features rarely found in single-binary tools.

### TurboQuant KV-Cache Compression
Memory bandwidth is the primary bottleneck for LLMs. Anvil features built-in **TurboQuant**, utilizing WHT-rotated low-bit quantization with backend-native kernels (Metal `TurboFlash`, CUDA, Vulkan, HIP). Enjoy up to ~4.3× KV-cache compression (e.g., `-type-k turbo4 -type-v turbo3`) allowing you to run massive context windows on consumer hardware without catastrophic performance degradation.

### Next-Generation Speculative Decoding
Stop waiting for tokens. Anvil integrates state-of-the-art speculative decoding techniques natively:
*   **MTP for Gemma 4:** Pair a Gemma 4 target model with its official MTP (Multi-Token Prediction) assistant head to achieve massive throughput gains on short-prompt tasks (up to +50% TPS).
*   **NextN for Qwen 3.6:** Utilize draft models that share the target's context via NextN, delivering massive speedups on Qwen 3.6 MoE and dense models.

### Seamless Open-Weight Ecosystem
Anvil speaks the language of the modern AI community natively:
*   **Hugging Face Integration:** Directly pull specific GGUF quants or use our interactive quant-picker to browse HF repositories.
*   **Ollama Compatibility:** Pull models straight from the Ollama registry. Anvil downloads the weights, extracts the chat template, and sets up your profile automatically.
*   **Resumable Networking:** Built-in robust download management ensures that multi-gigabyte models are downloaded safely, with automatic resuming for interrupted connections.

---

## Quickstart Guide

### 1. Installation

Our universal installer gets you running in seconds.

```bash
curl -fsSL https://anvil-llm.github.io/anvil/install.sh | sh
```

*Note on Linux x86_64 / NVIDIA:*
The installer is highly intelligent. On Linux, if it detects NVIDIA hardware, it automatically prefers our CUDA prebuilt binary. This binary **bundles the CUDA runtime**. This means you do **not** need to install the massive CUDA Toolkit; you only need the standard NVIDIA driver.

If you lack drivers, we provide a universal driver installer that works on any Linux distro:
```bash
curl -fsSL -O https://raw.githubusercontent.com/anvil-llm/anvil/main/docs/anvil-nvidia-install.sh
sudo sh anvil-nvidia-install.sh
```

### 2. Pulling Models

You can point Anvil directly at local files, or pull from the cloud:

```bash
# Pull and run directly from Ollama
anvil pull ollama:llama3.2:3b
anvil run llama3.2:3b

# Interactive pull from HuggingFace (browses available quants)
anvil pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF

# Direct pull of a specific HuggingFace quant
anvil pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

### 3. Running Models

```bash
# Run a local model (triggers TUI on first run)
anvil run ./models/my-custom-model.gguf
```

---

## Advanced Developer Usage

The CLI is a first-class citizen in Anvil. Once you understand your hardware, you can dictate exactly how the engine behaves.

### Maximizing Performance with CLI Flags

Override your persistent profiles to test extreme settings:

```bash
# Force massive context, unlimited GPU offload, TurboQuant, and Speculative Decoding
anvil run model.gguf \
  --ctx 128000 \
  --ngl -1 \
  --type-k turbo4 \
  --type-v turbo3 \
  --mtp
```

### Managing Persistent Profiles

If you find a CLI configuration you love, you can save it to the model's profile forever:

```bash
# Run with a custom temperature and save it as the new default for this model
anvil run llama3.2 --temp 0.85 --save
```

You can also manipulate profiles directly without running the model:

```bash
# View current settings
anvil profile llama3.2

# Explicitly set background parameters
anvil profile llama3.2 set n_ctx=131072 type_k=turbo4

# Remove a model and its profile
anvil rm llama3.2 --yes
```

---

## Building From Source

For contributors and power users, building Anvil is straightforward thanks to CMake.

```bash
# Clone the repository and its submodules
git clone --recursive https://github.com/anvil-llm/anvil
cd anvil

# Configure the build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Compile (adjust -j to your core count)
cmake --build build -j 8
```

---

## License

Anvil is released under the **MIT License**. We believe fundamental local AI tooling should be free, open, and unrestricted.

---
*The anvil doesn't ask questions. It just works.*