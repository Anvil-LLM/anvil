
#include "llama.h"
#include "ggml.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <clocale>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <algorithm>
#include <functional>
#include <mutex>
#ifdef __APPLE__
#include <IOKit/IOKitLib.h>
#include <sys/sysctl.h>
#include <mach-o/dyld.h>
#endif
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <io.h>
#else
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <glob.h>
#endif
#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi.h>
#endif
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>
#include "mdtty/mdtty.hpp"
#include <ftxui/dom/elements.hpp>
#include <iostream>
#include <ctime>
#include <cinttypes>
#include <map>
#include <nlohmann/json.hpp>
#include "httplib.h"
#include "chat.h"
#include <cxxmcp/client.hpp>

inline const char * ANVIL_LOGO = R"(
   ░███                          ░██░██
  ░██░██                            ░██
 ░██  ░██  ░████████  ░██    ░██ ░██░██
░█████████ ░██    ░██ ░██    ░██ ░██░██
░██    ░██ ░██    ░██  ░██  ░██  ░██░██
░██    ░██ ░██    ░██   ░██░██   ░██░██
░██    ░██ ░██    ░██    ░███    ░██░██
)";
inline const char * ANVIL_VERSION = "0.8.4";
inline const int    CONFIG_VERSION = 2;

inline volatile sig_atomic_t g_interrupted = 0;
inline void anvil_signal_handler(int) {
    g_interrupted = 1;
}

inline void install_sigint(void (*handler)(int), bool restart = true) {
#ifdef _WIN32
    signal(SIGINT, handler);
#else
    struct sigaction sa {};
    sa.sa_handler = handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = restart ? SA_RESTART : 0;
    sigaction(SIGINT, &sa, nullptr);
#endif
}

struct LlamaModel {
    llama_model * p = nullptr;
    explicit LlamaModel(llama_model * p_ = nullptr) : p(p_) {}
    ~LlamaModel() { if (p) llama_model_free(p); }
    LlamaModel(const LlamaModel &) = delete;
    LlamaModel & operator=(const LlamaModel &) = delete;
    LlamaModel(LlamaModel && o) noexcept : p(o.p) { o.p = nullptr; }
    LlamaModel & operator=(LlamaModel && o) noexcept {
        if (this != &o) { if (p) llama_model_free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    llama_model * get() const { return p; }
    operator llama_model * () const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

struct LlamaContext {
    llama_context * p = nullptr;
    explicit LlamaContext(llama_context * p_ = nullptr) : p(p_) {}
    ~LlamaContext() { if (p) llama_free(p); }
    LlamaContext(const LlamaContext &) = delete;
    LlamaContext & operator=(const LlamaContext &) = delete;
    LlamaContext(LlamaContext && o) noexcept : p(o.p) { o.p = nullptr; }
    LlamaContext & operator=(LlamaContext && o) noexcept {
        if (this != &o) { if (p) llama_free(p); p = o.p; o.p = nullptr; }
        return *this;
    }
    llama_context * get() const { return p; }
    operator llama_context * () const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

struct LlamaSampler {
    llama_sampler * p = nullptr;
    explicit LlamaSampler(llama_sampler * p_ = nullptr) : p(p_) {}
    ~LlamaSampler() { if (p) llama_sampler_free(p); }
    LlamaSampler(const LlamaSampler &) = delete;
    LlamaSampler & operator=(const LlamaSampler &) = delete;
    LlamaSampler(LlamaSampler && o) noexcept : p(o.p) { o.p = nullptr; }
    LlamaSampler & operator=(LlamaSampler && o) noexcept {
        if (this != &o) { if (p) llama_sampler_free(p); p = o.p; o.p = nullptr; }
        return *this;
    }

    void reset(llama_sampler * p_) {
        if (p_ != p) { if (p) llama_sampler_free(p); p = p_; }
    }
    llama_sampler * get() const { return p; }
    operator llama_sampler * () const { return p; }
    explicit operator bool() const { return p != nullptr; }
};

struct LlamaBackend {
    LlamaBackend() { llama_backend_init(); }
    ~LlamaBackend() { llama_backend_free(); }
};

inline bool parse_int(const std::string & s, int & out) {
    if (s.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (errno == ERANGE || end == s.c_str() || *end != '\0') return false;
    if (v < INT_MIN || v > INT_MAX) return false;
    out = static_cast<int>(v);
    return true;
}

inline bool parse_float(const std::string & s, float & out) {
    if (s.empty()) return false;
    errno = 0;
    char * end = nullptr;
    const float v = std::strtof(s.c_str(), &end);
    if (errno == ERANGE || end == s.c_str() || *end != '\0') return false;
    if (!std::isfinite(v)) return false;
    out = v;
    return true;
}

inline bool parse_uint64(const std::string & s, uint64_t & out) {
    if (s.empty() || s[0] == '-') return false;
    errno = 0;
    char * end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno == ERANGE || end == s.c_str() || *end != '\0') return false;
    out = static_cast<uint64_t>(v);
    return true;
}

struct KVTypeOption {
    const char * label;
    ggml_type    type;
    const char * short_name;
};

inline const KVTypeOption KV_OPTIONS[] = {
    { "f16    (no compression)",   GGML_TYPE_F16,       "f16"    },
    { "q8_0   (8-bit, lossless)",  GGML_TYPE_Q8_0,      "q8_0"   },
    { "turbo4 (TurboQuant 4-bit)", GGML_TYPE_TURBO4_0,  "turbo4" },
    { "turbo3 (TurboQuant 3-bit)", GGML_TYPE_TURBO3_0,  "turbo3" },
    { "turbo2 (TurboQuant 2-bit)", GGML_TYPE_TURBO2_0,  "turbo2" },
};

inline constexpr int KV_OPTIONS_COUNT = static_cast<int>(sizeof(KV_OPTIONS) / sizeof(KV_OPTIONS[0]));

struct KVPreset {
    const char * label;
    int k_idx;
    int v_idx;
};

inline const KVPreset KV_PRESETS[] = {
    { "Recommended   (K=turbo4, V=turbo3)  4.2x  ~1% loss",   2, 3 },
    { "Quality+      (K=q8_0,   V=turbo3)  ~3x   <1% loss",   1, 3 },
    { "Max Compress  (K=turbo4, V=turbo2)  6.1x  ~3% loss",   2, 4 },
    { "No Compress   (K=f16,    V=f16)     1x    baseline",    0, 0 },
    { "Custom...",                                              -1, -1 },
};

inline constexpr int KV_PRESETS_COUNT = static_cast<int>(sizeof(KV_PRESETS) / sizeof(KV_PRESETS[0]));

struct GPUInfo {
    std::string name;
    std::string vendor;
    uint64_t    vram_mb = 0;
    bool        is_discrete = false;
};

struct HWInfo {
    std::string os;
    std::string arch;
    std::string cpu;
    uint64_t    ram_bytes = 0;
    int         cpu_threads = 0;
    std::vector<GPUInfo> gpus;
    bool        apple_silicon = false;
};

struct AnvilConfig {
    int       version    = CONFIG_VERSION;
    int       ngl        = -1;
    int       n_ctx      = 0;
    int       n_threads  = 0;
    float     temp       = 0.8f;
    int       top_k      = 40;
    float     top_p      = 0.95f;
    float     repeat_penalty = 1.1f;
    float     min_p        = 0.05f;
    int       repeat_last_n = 64;
    float     typical      = 1.0f;
    int       mirostat     = 0;
    float     mirostat_lr  = 0.1f;
    float     mirostat_ent = 5.0f;
    bool      ignore_eos   = false;
    int       n_batch      = 0;
    std::string samplers;
    bool      flash_attn = true;
    bool      mtp        = false;
    int       seed       = -1;
    ggml_type type_k     = GGML_TYPE_Q8_0;
    ggml_type type_v     = GGML_TYPE_TURBO3_0;
    std::string model;
    std::string system_prompt;
};

inline std::string config_dir() {
    const char * home = std::getenv("HOME");
#ifdef _WIN32
    if (!home || !home[0]) home = std::getenv("USERPROFILE");
#endif
    if (!home || !home[0]) home = ".";
    return std::string(home) + "/.anvil";
}

inline std::string config_path() {
    return config_dir() + "/config.json";
}

inline std::string sessions_dir() {
    return config_dir() + "/sessions";
}

struct Utf8Buffer {
    std::string pending;

    std::string feed(const std::string & chunk) {
        pending += chunk;
        size_t safe = pending.size();
        if (safe == 0) return "";
        size_t i = safe - 1;
        while (i > 0 && (pending[i] & 0xC0) == 0x80) i--;
        const unsigned char lead = static_cast<unsigned char>(pending[i]);
        int expected = 1;
        if      ((lead & 0x80) == 0x00) expected = 1;
        else if ((lead & 0xE0) == 0xC0) expected = 2;
        else if ((lead & 0xF0) == 0xE0) expected = 3;
        else if ((lead & 0xF8) == 0xF0) expected = 4;
        const int actual = static_cast<int>(safe - i);
        if (actual < expected) safe = i;
        std::string out = pending.substr(0, safe);
        pending = pending.substr(safe);
        return out;
    }

    std::string flush() {
        std::string out = pending;
        pending.clear();
        return out;
    }
};

struct MarkdownStream {
    mdtty::Renderer resp_s;
    mdtty::Renderer think_s;

    std::string pending;
    bool color = false;
    bool in_think_s = false;

    static void sink_out(std::string_view s) {
        std::fwrite(s.data(), 1, s.size(), stdout);
        fflush(stdout);
    }

    MarkdownStream()
        : resp_s(sink_out, {}), think_s(sink_out, think_config()),
          color(mdtty::Renderer::is_tty()) {}

    static mdtty::Config think_config() {
        mdtty::Config tc;
        tc.reset = "\033[0;2m";
        return tc;
    }

    static bool tag_at(const std::string & t, size_t i, bool closing, size_t & len) {
        static const char * open_[]  = { "<thinking>", "<think>", "〈thinking〉", "〈think〉" };
        static const char * close_[] = { "</thinking>", "</think>", "〈/thinking〉", "〈/think〉" };
        const char * const * tags = closing ? close_ : open_;
        for (size_t k = 0; k < 4; k++) {
            const size_t n = std::strlen(tags[k]);
            if (i + n <= t.size() && std::strncmp(tags[k], t.c_str() + i, n) == 0) {
                len = n;
                return true;
            }
        }
        return false;
    }

    static bool is_tag_prefix(const std::string & s) {
        static const char * tags[] = {
            "<thinking>", "<think>", "〈thinking〉", "〈think〉",
            "</thinking>", "</think>", "〈/thinking〉", "〈/think〉",
        };
        for (const char * tag : tags) {
            const size_t n = std::strlen(tag);
            if (n > s.size() && std::strncmp(tag, s.c_str(), s.size()) == 0) return true;
        }
        return false;
    }

    static size_t partial_suffix(const std::string & b) {
        static constexpr size_t HOLD_MAX = 15;
        const size_t lim = std::min(b.size(), HOLD_MAX);
        for (size_t L = lim; L >= 1; --L) {
            if (is_tag_prefix(b.substr(b.size() - L))) return L;
        }
        return 0;
    }

    void drain_s(bool force = false) {
        while (!pending.empty()) {
            size_t tag_i = std::string::npos;
            size_t tag_len = 0;
            bool   closing = false;
            for (size_t i = 0; i < pending.size(); ++i) {
                size_t len = 0;
                if (tag_at(pending, i, true, len)) {
                    tag_i = i; tag_len = len; closing = true;
                    break;
                }
                if (!in_think_s && tag_at(pending, i, false, len)) {
                    tag_i = i; tag_len = len;
                    break;
                }
            }
            if (tag_i != std::string::npos) {
                if (tag_i > 0) (in_think_s ? think_s : resp_s).feed(pending.substr(0, tag_i));
                (in_think_s ? think_s : resp_s).flush();
                if (closing) {
                    if (in_think_s) {
                        printf(color ? "\033[2;90m└─ /Thinking ─\033[0m\n" : "└─ /Thinking ─\n");
                        in_think_s = false;
                    }
                } else if (!in_think_s) {
                    printf(color ? "\033[2;90m┌─ Thinking ─\033[0m\n" : "┌─ Thinking ─\n");
                    in_think_s = true;
                }
                pending.erase(0, tag_i + tag_len);
                continue;
            }
            const size_t hold = force ? 0 : partial_suffix(pending);
            const size_t emit = pending.size() - hold;
            if (emit == 0) break;
            (in_think_s ? think_s : resp_s).feed(pending.substr(0, emit));
            pending.erase(0, emit);
        }
    }

    void feed(const std::string & s) {
        pending += s;
        drain_s(false);
    }

    void flush() {
        drain_s(true);
        (in_think_s ? think_s : resp_s).flush();
        if (in_think_s) {
            printf(color ? "\033[2;90m└─ /Thinking ─\033[0m\n" : "└─ /Thinking ─\n");
            in_think_s = false;
        }
        resp_s.reset();
        think_s.reset();
        pending.clear();
    }
};

struct ChatMessage {
    std::string role;
    std::string content;
    std::string image_path;
    std::string tool_calls_json;
    std::string tool_call_id;
};

struct GenStats {
    int    tokens_generated = 0;
    double elapsed_sec      = 0.0;

    double tps() const {
        return elapsed_sec > 0.0 ? tokens_generated / elapsed_sec : 0.0;
    }
};

inline bool stdin_is_tty() {
#ifdef _WIN32
    return _isatty(_fileno(stdin)) != 0;
#else
    return isatty(STDIN_FILENO) != 0;
#endif
}

struct CliArgs {
    std::string sub;
    std::vector<std::string> sub_args;
    std::string model;
    std::string friendly;
    int         n_ctx       = 0;
    int         ngl         = -1;
    int         n_threads   = 0;
    float       temp        = -1.0f;
    int         top_k       = -1;
    float       top_p       = -1.0f;
    float       repeat_penalty = -1.0f;
    bool        flash_attn  = false;
    bool        no_flash_attn = false;
    bool        mtp         = false;
    bool        save_profile = false;
    bool        help        = false;
    bool        invalid     = false;
    bool        version     = false;
    bool        setup       = false;
    bool        resume      = false;
    bool        non_interactive = false;
    std::string type_k;
    std::string type_v;
    std::string system_prompt;
    std::string prompt;
    std::string grammar;
    std::string mmproj;
    std::string rag_dir;
    std::string image;
    std::string token;
    std::string api_key;
    int         max_tokens  = -1;
    int         port        = 8080;
    std::string host        = "127.0.0.1";
    int         bench_tokens = 32;
    int         bench_ctx   = 0;
    int         slots       = 1;
    int         seed        = -1;
    int         n_batch     = 0;
    int         n_ubatch    = 0;
    float       min_p       = -1.0f;
    int         repeat_last_n = -1;
    float       typical     = -1.0f;
    int         mirostat    = -1;
    float       mirostat_lr = -1.0f;
    float       mirostat_ent = -1.0f;
    bool        ignore_eos  = false;
    bool        use_mlock   = false;
    bool        no_mmap     = false;
    bool        interactive = false;
    bool        verbose     = false;
    float       rope_freq_base  = 0.0f;
    float       rope_freq_scale = 0.0f;
    std::string rope_scaling;
    std::string samplers;
    std::string file_prompt;
};

inline constexpr int   MAX_CTX      = INT32_MAX;
inline constexpr int   MAX_THREADS  = 1024;
inline constexpr float MAX_TEMP     = 5.0f;
inline constexpr int   MAX_TOP_K    = 100000;

void print_usage();
CliArgs parse_args(int argc, char ** argv);

void print_usage() {
    printf("anvil %s — Forge anything.\n\n", ANVIL_VERSION);
    printf("Usage:\n");
    printf("  anvil run <model> [options]     Run a model with chat REPL\n");
    printf("  anvil run <model> -p \"prompt\"   Single-shot generation\n");
    printf("  anvil <model> [options]         Run (model name or file path)\n");
    printf("  anvil models                    List registered models\n");
    printf("  anvil models import <file.gguf> [--name <name>]\n");
    printf("  anvil profile <name>            Show a model's persistent profile\n");
    printf("  anvil profile <name> set k=v .. Save profile settings\n");
    printf("  anvil rm <name> [--yes]         Unregister a model\n");
    printf("  anvil pull ollama:<name>[:tag]  Pull from the Ollama registry\n");
    printf("  anvil pull hf:<repo>[:file]     Pull from HuggingFace\n");
    printf("  anvil serve [--port <n>]        OpenAI-compatible API server\n");
    printf("                         [--slots <n>]  KV slots with cross-request prefix caching\n");
    printf("  anvil bench <model> [options]   TurboQuant KV benchmark\n");
    printf("  anvil doctor                    System diagnostics\n");
    printf("  anvil self-update               Update to the latest release\n");
    printf("  anvil --help                    Show this help\n");
    printf("  anvil --version                 Show version\n");
    printf("  anvil --setup                   Re-run hardware setup TUI\n\n");
    printf("Options:\n");
    printf("  -c, --ctx <n>            Context size (default: auto from model)\n");
    printf("  -ngl, --n-gpu-layers <n> GPU layers to offload (default: auto)\n");
    printf("  -t, --temp <f>           Sampling temperature (default: 0.8)\n");
    printf("  --top-k <n>              Top-k sampling (default: 40, 0 = off)\n");
    printf("  --top-p <f>              Top-p (nucleus) sampling (default: 0.95, 1 = off)\n");
    printf("  --repeat-penalty <f>     Token repeat penalty (default: 1.1, 1 = off)\n");
    printf("  --threads <n>            CPU threads (default: auto)\n");
    printf("  --flash-attn             Enable flash attention (default: on)\n");
    printf("  --no-flash-attn          Disable flash attention\n");
    printf("  --type-k <type>          K cache type: f16|q8_0|turbo4|turbo3|turbo2\n");
    printf("  --type-v <type>          V cache type: f16|q8_0|turbo4|turbo3|turbo2\n");
    printf("  --mtp                    Enable MTP speculative decoding\n");
    printf("  --grammar <file>         GBNF grammar file for constrained output\n");
    printf("  -s, --system <text>      System prompt\n");
    printf("  -p, --prompt <text>      User prompt (non-interactive mode)\n");
    printf("  -n, --max-tokens <n>     Max tokens to generate (default: unlimited)\n");
    printf("      --mmproj <file>       Vision projector (multimodal models)\n");
    printf("      --image <file>        Attach an image (single-shot mode)\n");
    printf("      --rag <dir>           Retrieve context from a folder of documents\n");
    printf("      --resume              Resume the latest session for this model\n");
    printf("      --token <tok>         HF token for gated repos (or HF_TOKEN env)\n");
    printf("      --save                Persist CLI overrides into the model's profile\n\n");
    printf("llama.cpp compatibility (upstream CLI flags accepted by 'anvil run'):\n");
    printf("  -m, --model <path>       Model (same as positional arg)\n");
    printf("  -f, --file <file>        Read the prompt from a file (overrides -p)\n");
    printf("      --ctx-size <n>       Alias for --ctx\n");
    printf("      --n-predict <n>      Alias for -n/--max-tokens (also --predict)\n");
    printf("  -b, --batch-size <n>     Batch size (default: = ctx size)\n");
    printf("      --seed <n>           RNG seed (default: random)\n");
    printf("      --min-p <f>          Min-p sampling (default: 0.05, 0 = off)\n");
    printf("      --repeat-last-n <n>  Last N tokens to penalize (default: 64, 0 = off)\n");
    printf("      --typical <f>        Typical sampling (default: 1, 1 = off)\n");
    printf("      --mirostat <0|1|2>   Mirostat sampling mode\n");
    printf("      --mirostat-lr <f>    Mirostat learning rate (default: 0.1)\n");
    printf("      --mirostat-ent <f>   Mirostat target entropy (default: 5.0)\n");
    printf("      --ignore-eos         Do not stop generation at EOS\n");
    printf("      --mlock              Lock the model in RAM\n");
    printf("      --no-mmap            Disable memory-mapped model loading\n");
    printf("  -i, --interactive        Force interactive REPL after a -p prompt\n");
    printf("  -cnv, --conversation     Same as --interactive\n");
    printf("      --rope-scaling <t>   none|linear|yarn|longrope\n");
    printf("      --rope-freq-base <f> RoPE base frequency override\n");
    printf("      --rope-freq-scale <f> RoPE frequency scaling override\n");
    printf("      --samplers <list>    Exact chain: penalties;top_k;typical;top_p;min_p;temp;mirostat_v2;dist\n");
    printf("      --cache-type-k <t>   Alias for --type-k\n");
    printf("      --cache-type-v <t>   Alias for --type-v\n");
    printf("  Note: anvil keeps -t = --temp and -s = --system (llama.cpp uses -t for threads\n");
    printf("  and -s for seed). Use --threads and --seed for those. -fa works like llama.cpp.\n\n");
    printf("Model registry:\n");
    printf("  <model> may be a friendly name (registered via pull/import) or a file path.\n");
    printf("  Per-model settings live in ~/.anvil/models.json (profiles).\n");
    printf("  Precedence: CLI flags > model profile > global config.\n\n");
    printf("TurboQuant KV presets (from setup TUI):\n");
    printf("  Recommended:  K=turbo4 V=turbo3  (4.2x, ~1%% quality loss)\n");
    printf("  Quality+:     K=q8_0   V=turbo3  (~3x,  <1%% quality loss)\n");
    printf("  High Compression: K=turbo4 V=turbo2  (6.1x, ~3%% quality loss)\n\n");
    printf("Examples:\n");
    printf("  anvil run model.gguf\n");
    printf("  anvil run model.gguf --ctx 131072 --type-k q8_0 --type-v turbo3\n");
    printf("  anvil run model.gguf -p \"Explain quantum computing\" -n 200\n");
    printf("  anvil run model.gguf --grammar json.gbnf -p \"List 3 colors\"\n");
}

static bool set_int(const std::string & arg_name, const std::string & value,
                    int min, int max, int & out, CliArgs & a) {
    int v = 0;
    if (!parse_int(value, v) || v < min || v > max) {
        fprintf(stderr, "error: invalid value for %s: '%s' (expected %d..%d)\n",
                arg_name.c_str(), value.c_str(), min, max);
        a.invalid = true;
        a.help = true;
        return false;
    }
    out = v;
    return true;
}

static bool set_float(const std::string & arg_name, const std::string & value,
                      float min, float max, float & out, CliArgs & a) {
    float v = 0.0f;
    if (!parse_float(value, v) || v < min || v > max) {
        fprintf(stderr, "error: invalid value for %s: '%s' (expected %.2f..%.2f)\n",
                arg_name.c_str(), value.c_str(), min, max);
        a.invalid = true;
        a.help = true;
        return false;
    }
    out = v;
    return true;
}

CliArgs parse_args(int argc, char ** argv) {
    CliArgs a;

    if (argc < 2) { a.invalid = true; a.help = true; return a; }

    const std::string first = argv[1];
    if (first == "models" || first == "profile" || first == "rm" || first == "pull" ||
        first == "serve" || first == "doctor" || first == "self-update" || first == "bench" ||
        first == "mcp" || first == "keys" || first == "eval") {
        a.sub = first;
        for (int i = 2; i < argc; i++) a.sub_args.emplace_back(argv[i]);
        return a;
    }
    int i = 1;
    if (i < argc && std::string(argv[i]) == "run") i++;
    for (; i < argc; i++) {
        const std::string arg = argv[i];
        if      (arg == "--help" || arg == "-h")                          { a.help = true; }
        else if (arg == "--version")                                      { a.version = true; }
        else if (arg == "--setup")                                        { a.setup = true; }
        else if ((arg == "-c" || arg == "--ctx" || arg == "--ctx-size") && i + 1 < argc) {
            set_int(arg, argv[++i], 1, MAX_CTX, a.n_ctx, a);
        }
        else if ((arg == "-ngl" || arg == "--ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {

            set_int(arg, argv[++i], -1, 10000, a.ngl, a);
        }
        else if (arg == "-t" && i + 1 < argc) {
            const std::string val = argv[i + 1];
            float f = 0.0f;
            int n = 0;
            if (parse_float(val, f) && f >= 0.0f && f <= MAX_TEMP) {
                a.temp = f;
                i++;
            } else if (parse_int(val, n) && n >= 1) {
                a.n_threads = n;
                i++;
            } else {
                set_float("-t", val, 0.0f, MAX_TEMP, a.temp, a);
            }
        }
        else if (arg == "--temp" && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, MAX_TEMP, a.temp, a);
        }
        else if (arg == "--top-k" && i + 1 < argc) {
            set_int(arg, argv[++i], 0, MAX_TOP_K, a.top_k, a);
        }
        else if (arg == "--top-p" && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, 1.0f, a.top_p, a);
        }
        else if (arg == "--repeat-penalty" && i + 1 < argc) {
            set_float(arg, argv[++i], 1.0f, 100.0f, a.repeat_penalty, a);
        }
        else if (arg == "--threads" && i + 1 < argc) {
            set_int(arg, argv[++i], 1, MAX_THREADS, a.n_threads, a);
        }
        else if (arg == "--flash-attn" || arg == "-fa")                   { a.flash_attn = true; }
        else if (arg == "--no-flash-attn")                                { a.no_flash_attn = true; a.flash_attn = false; }
        else if (arg == "--type-k" && i + 1 < argc)                       { a.type_k = argv[++i]; }
        else if (arg == "--type-v" && i + 1 < argc)                       { a.type_v = argv[++i]; }
        else if (arg == "--mtp")                                          { a.mtp = true; }
        else if (arg == "--save")                                         { a.save_profile = true; }
        else if (arg == "--grammar" && i + 1 < argc)                      { a.grammar = argv[++i]; }
        else if ((arg == "-s" || arg == "--system") && i + 1 < argc)      { a.system_prompt = argv[++i]; }
        else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc)      { a.prompt = argv[++i]; }
        else if ((arg == "-n" || arg == "--max-tokens" || arg == "--n-predict" || arg == "--predict") && i + 1 < argc) {
            set_int(arg, argv[++i], -1, INT32_MAX, a.max_tokens, a);
        }
        else if (arg == "--mmproj" && i + 1 < argc)                       { a.mmproj = argv[++i]; }
        else if (arg == "--rag" && i + 1 < argc)                          { a.rag_dir = argv[++i]; }
        else if (arg == "--resume")                                       { a.resume = true; }
        else if (arg == "--token" && i + 1 < argc)                        { a.token = argv[++i]; }
        else if (arg == "--image" && i + 1 < argc)                        { a.image = argv[++i]; }
        else if (arg == "--port" && i + 1 < argc) {
            set_int(arg, argv[++i], 1, 65535, a.port, a);
        }
        else if (arg == "--host" && i + 1 < argc)                         { a.host = argv[++i]; }
        else if (arg == "--bench-tokens" && i + 1 < argc) {
            set_int(arg, argv[++i], 1, 1000000, a.bench_tokens, a);
        }
        else if (arg == "--bench-ctx" && i + 1 < argc) {
            set_int(arg, argv[++i], 1, MAX_CTX, a.bench_ctx, a);
        }
        else if ((arg == "-m" || arg == "--model") && i + 1 < argc)       { a.model = argv[++i]; }
        else if ((arg == "-f" || arg == "--file") && i + 1 < argc)        { a.file_prompt = argv[++i]; }
        else if ((arg == "-b" || arg == "--batch-size") && i + 1 < argc) {
            set_int(arg, argv[++i], 1, 1000000, a.n_batch, a);
        }
        else if (arg == "--ubatch-size" && i + 1 < argc) {
            set_int(arg, argv[++i], 1, 1000000, a.n_ubatch, a);
        }
        else if (arg == "--seed" && i + 1 < argc) {
            set_int(arg, argv[++i], -1, INT32_MAX, a.seed, a);
        }
        else if (arg == "--min-p" && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, 1.0f, a.min_p, a);
        }
        else if (arg == "--repeat-last-n" && i + 1 < argc) {
            set_int(arg, argv[++i], -1, 1000000, a.repeat_last_n, a);
        }
        else if (arg == "--typical" && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, 1.0f, a.typical, a);
        }
        else if (arg == "--mirostat" && i + 1 < argc) {
            set_int(arg, argv[++i], 0, 2, a.mirostat, a);
        }
        else if (arg == "--mirostat-lr" && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, 1.0f, a.mirostat_lr, a);
        }
        else if (arg == "--mirostat-ent" && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, 100.0f, a.mirostat_ent, a);
        }
        else if (arg == "--json-schema") {
            fprintf(stderr, "error: --json-schema is not supported yet; use --grammar <file> with a GBNF grammar\n");
            a.invalid = true;
            a.help = true;
        }
        else if (arg == "--ignore-eos")                                    { a.ignore_eos = true; }
        else if (arg == "--mlock")                                         { a.use_mlock = true; }
        else if (arg == "--no-mmap")                                       { a.no_mmap = true; }
        else if ((arg == "-i" || arg == "--interactive" || arg == "-cnv" || arg == "--conversation")) {
            a.interactive = true;
        }
        else if (arg == "-v" || arg == "--verbose")                        { a.verbose = true; }
        else if (arg == "--cache-type-k" && i + 1 < argc)                  { a.type_k = argv[++i]; }
        else if (arg == "--cache-type-v" && i + 1 < argc)                  { a.type_v = argv[++i]; }
        else if (arg == "--rope-scaling" && i + 1 < argc)                  { a.rope_scaling = argv[++i]; }
        else if (arg == "--rope-freq-base" && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, 1000000.0f, a.rope_freq_base, a);
        }
        else if (arg == "--rope-freq-scale" && i + 1 < argc) {
            set_float(arg, argv[++i], 0.0f, 10.0f, a.rope_freq_scale, a);
        }
        else if ((arg == "--samplers" || arg == "--sampling-seq") && i + 1 < argc) {
            a.samplers = argv[++i];
        }
        else if (arg[0] != '-')                                           { a.model = arg; }
        else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            a.invalid = true;
            a.help = true;
        }
    }
    return a;
}

ggml_type kv_type_from_name(const std::string & name);
const char * kv_type_short(ggml_type type);

void write_config(const AnvilConfig & cfg);
AnvilConfig load_config();
bool config_exists();

ggml_type kv_type_from_name(const std::string & name) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (name == KV_OPTIONS[i].short_name) return KV_OPTIONS[i].type;
    }
    fprintf(stderr, "\033[33mwarning: unknown kv type '%s', using f16\033[0m\n", name.c_str());
    return GGML_TYPE_F16;
}

const char * kv_type_short(ggml_type type) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (KV_OPTIONS[i].type == type) return KV_OPTIONS[i].short_name;
    }
    return "f16";
}

static std::string json_get(const std::string & json, const std::string & key) {
    const std::string search = "\"" + key + "\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) pos++;
    if (pos >= json.size()) return "";
    if (json[pos] == '"') {
        pos++;
        std::string result;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                pos++;
                switch (json[pos]) {
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    case 'u': {
                        if (pos + 4 < json.size()) {
                            int cp = 0;
                            for (int j = 0; j < 4; j++) {
                                pos++;
                                cp <<= 4;
                                const char h = json[pos];
                                if (h >= '0' && h <= '9')      cp |= h - '0';
                                else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                                else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                            }
                            if (cp <= 0x7F) {
                                result += static_cast<char>(cp);
                            } else if (cp <= 0x7FF) {
                                result += static_cast<char>(0xC0 | (cp >> 6));
                                result += static_cast<char>(0x80 | (cp & 0x3F));
                            } else {
                                result += static_cast<char>(0xE0 | (cp >> 12));
                                result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                result += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                        }
                        break;
                    }
                    default: result += json[pos]; break;
                }
            } else {
                result += json[pos];
            }
            pos++;
        }
        return result;
    }
    auto end = pos;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != '\n' && json[end] != '\r') end++;
    std::string val = json.substr(pos, end - pos);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.pop_back();
    return val;
}

static std::string json_escape(const std::string & s) {
    std::string r;
    for (const char c : s) {
        switch (c) {
            case '\\': r += "\\\\"; break;
            case '"':  r += "\\\""; break;
            case '\n': r += "\\n";  break;
            case '\r': r += "\\r";  break;
            case '\t': r += "\\t";  break;
            default:   r += c;      break;
        }
    }
    return r;
}

void write_config(const AnvilConfig & cfg) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(config_dir(), ec);
    if (ec) {
        fprintf(stderr, "\033[33mwarning: could not create config dir %s: %s\033[0m\n",
                config_dir().c_str(), ec.message().c_str());
    }

    const std::string tmp = config_path() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) {
            fprintf(stderr, "\033[33mwarning: could not write config to %s\033[0m\n", tmp.c_str());
            return;
        }
        f << "{\n";
        f << "  \"version\": " << cfg.version << ",\n";
        f << "  \"ngl\": " << cfg.ngl << ",\n";
        f << "  \"n_ctx\": " << cfg.n_ctx << ",\n";
        f << "  \"n_threads\": " << cfg.n_threads << ",\n";
        f << "  \"temp\": " << cfg.temp << ",\n";
        f << "  \"top_k\": " << cfg.top_k << ",\n";
        f << "  \"top_p\": " << cfg.top_p << ",\n";
        f << "  \"repeat_penalty\": " << cfg.repeat_penalty << ",\n";
        f << "  \"flash_attn\": " << (cfg.flash_attn ? "true" : "false") << ",\n";
        f << "  \"mtp\": " << (cfg.mtp ? "true" : "false") << ",\n";
        f << "  \"type_k\": \"" << kv_type_short(cfg.type_k) << "\",\n";
        f << "  \"type_v\": \"" << kv_type_short(cfg.type_v) << "\",\n";
        f << "  \"model\": \"" << json_escape(cfg.model) << "\"\n";
        f << "}\n";
        f.flush();
        if (!f) {
            fprintf(stderr, "\033[33mwarning: failed writing config to %s\033[0m\n", tmp.c_str());
            return;
        }
    }

    fs::rename(tmp, config_path(), ec);
    if (ec) {
        fprintf(stderr, "\033[33mwarning: could not rename config into place at %s: %s\033[0m\n",
                config_path().c_str(), ec.message().c_str());
    }
}

static bool json_get_bool(const std::string & json, const std::string & key, bool def) {
    const std::string s = json_get(json, key);
    return s.empty() ? def : (s == "true");
}

AnvilConfig load_config() {
    AnvilConfig cfg;
    std::ifstream f(config_path());
    if (!f) return cfg;
    const std::string json((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    std::string s;
    int tmp_int = 0;
    float tmp_float = 0.0f;

    s = json_get(json, "version");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.version = tmp_int;
    s = json_get(json, "ngl");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.ngl = tmp_int;
    s = json_get(json, "n_ctx");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.n_ctx = tmp_int;
    s = json_get(json, "n_threads");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.n_threads = tmp_int;
    s = json_get(json, "temp");
    if (!s.empty() && parse_float(s, tmp_float)) cfg.temp = tmp_float;
    s = json_get(json, "top_k");
    if (!s.empty() && parse_int(s, tmp_int)) cfg.top_k = tmp_int;
    s = json_get(json, "top_p");
    if (!s.empty() && parse_float(s, tmp_float)) cfg.top_p = tmp_float;
    s = json_get(json, "repeat_penalty");
    if (!s.empty() && parse_float(s, tmp_float)) cfg.repeat_penalty = tmp_float;

    cfg.flash_attn = json_get_bool(json, "flash_attn", cfg.flash_attn);
    cfg.mtp        = json_get_bool(json, "mtp", cfg.mtp);

    s = json_get(json, "type_k");
    if (!s.empty()) cfg.type_k = kv_type_from_name(s);
    s = json_get(json, "type_v");
    if (!s.empty()) cfg.type_v = kv_type_from_name(s);
    s = json_get(json, "model");
    if (!s.empty()) cfg.model = s;

    if (cfg.version < 2) {
        const std::string no_turbo = json_get(json, "no_turbo");
        if (no_turbo == "true") {
            cfg.type_k = GGML_TYPE_F16;
            cfg.type_v = GGML_TYPE_F16;
        } else {
            cfg.type_k = GGML_TYPE_Q8_0;
            cfg.type_v = GGML_TYPE_TURBO3_0;
        }
        cfg.version = CONFIG_VERSION;
    }
    return cfg;
}

bool config_exists() {
    std::ifstream f(config_path());
    return f.good();
}

inline std::string models_json_path() { return config_dir() + "/models.json"; }
inline std::string models_dir()       { return config_dir() + "/models"; }

bool validate_gguf(const std::string & path);
std::string gguf_check_error(const std::string & path);

struct ModelProfile {

    std::map<std::string, nlohmann::json> settings;

    nlohmann::json to_json() const { return nlohmann::json(settings); }

    static ModelProfile from_json(const nlohmann::json & j) {
        ModelProfile p;
        if (j.is_object()) {
            for (auto it = j.begin(); it != j.end(); ++it) p.settings[it.key()] = it.value();
        }
        return p;
    }

    bool empty() const { return settings.empty(); }
    void clear()       { settings.clear(); }
    void set(const std::string & k, const nlohmann::json & v) { settings[k] = v; }
    bool unset(const std::string & k) { return settings.erase(k) > 0; }

    int apply_to(AnvilConfig & cfg) const {
        int n = 0;
        for (const auto & [k, v] : settings) {
            try {
                if      (k == "n_ctx")          { cfg.n_ctx = v.get<int>();        n++; }
                else if (k == "ngl")            { cfg.ngl = v.get<int>();          n++; }
                else if (k == "n_threads")      { cfg.n_threads = v.get<int>();    n++; }
                else if (k == "temp")           { cfg.temp = v.get<float>();       n++; }
                else if (k == "top_k")          { cfg.top_k = v.get<int>();        n++; }
                else if (k == "top_p")          { cfg.top_p = v.get<float>();      n++; }
                else if (k == "repeat_penalty") { cfg.repeat_penalty = v.get<float>(); n++; }
                else if (k == "flash_attn")     { cfg.flash_attn = v.get<bool>();  n++; }
                else if (k == "mtp")            { cfg.mtp = v.get<bool>();         n++; }
                else if (k == "type_k")         { cfg.type_k = kv_type_from_name(v.get<std::string>()); n++; }
                else if (k == "type_v")         { cfg.type_v = kv_type_from_name(v.get<std::string>()); n++; }
                else if (k == "seed")           { cfg.seed = v.get<int>();          n++; }
                else if (k == "min_p")          { cfg.min_p = v.get<float>();       n++; }
                else if (k == "repeat_last_n")  { cfg.repeat_last_n = v.get<int>(); n++; }
                else if (k == "typical")        { cfg.typical = v.get<float>();     n++; }
                else if (k == "mirostat")       { cfg.mirostat = v.get<int>();      n++; }
                else if (k == "mirostat_lr")    { cfg.mirostat_lr = v.get<float>(); n++; }
                else if (k == "mirostat_ent")   { cfg.mirostat_ent = v.get<float>(); n++; }
                else if (k == "ignore_eos")     { cfg.ignore_eos = v.get<bool>();   n++; }
                else if (k == "n_batch")        { cfg.n_batch = v.get<int>();       n++; }
                else if (k == "samplers")       { cfg.samplers = v.get<std::string>(); n++; }
                else if (k == "system_prompt")  { cfg.system_prompt = v.get<std::string>(); n++; }
            } catch (const nlohmann::json::exception &) {
                fprintf(stderr, "\033[33mwarning: skipping invalid profile key '%s' (wrong type)\033[0m\n", k.c_str());
            }
        }
        return n;
    }

    static bool valid_key(const std::string & k) {
        static const char * keys[] = {
            "n_ctx", "ngl", "n_threads", "temp", "top_k", "top_p",
            "repeat_penalty", "flash_attn", "mtp", "type_k", "type_v", "system_prompt",
            "seed", "min_p", "repeat_last_n", "typical", "mirostat", "mirostat_lr",
            "mirostat_ent", "ignore_eos", "n_batch", "samplers"
        };
        for (const char * kk : keys) if (k == kk) return true;
        return false;
    }
};

struct ModelEntry {
    std::string name;
    std::string path;
    std::string source = "local";
    std::string source_id;
    uint64_t    size_bytes = 0;
    std::string added;
    std::string desc;
    int64_t     trained_ctx = 0;
    std::string chat_template;
    std::string license;
    std::string params_json;
    nlohmann::json gguf_meta;
    ModelProfile profile;
    bool         seeded = false;

    nlohmann::json to_json() const {
        return {
            {"name", name}, {"path", path}, {"source", source},
            {"source_id", source_id}, {"size_bytes", size_bytes}, {"added", added},
            {"desc", desc}, {"trained_ctx", trained_ctx},
            {"chat_template", chat_template}, {"license", license},
            {"params_json", params_json}, {"gguf_meta", gguf_meta},
            {"settings", profile.to_json()}, {"seeded", seeded},
        };
    }

    static ModelEntry from_json(const nlohmann::json & j) {
        ModelEntry e;
        auto get = [&](const char * k, auto & out) {
            if (j.contains(k)) {
                try { out = j[k].get<std::decay_t<decltype(out)>>(); }
                catch (const nlohmann::json::exception &) {}
            }
        };
        get("name", e.name); get("path", e.path); get("source", e.source);
        get("source_id", e.source_id); get("size_bytes", e.size_bytes);
        get("added", e.added); get("desc", e.desc); get("trained_ctx", e.trained_ctx);
        get("chat_template", e.chat_template); get("license", e.license);
        get("params_json", e.params_json); get("seeded", e.seeded);
        if (j.contains("gguf_meta") && j["gguf_meta"].is_object()) e.gguf_meta = j["gguf_meta"];
        if (j.contains("settings")) e.profile = ModelProfile::from_json(j["settings"]);
        return e;
    }
};

static bool seed_model_profile(ModelProfile & p, int64_t trained_ctx,
                               const nlohmann::json & gguf_meta) {
    bool changed = false;
    auto fill = [&](const char * k, nlohmann::json v) {
        if (p.settings.count(k)) return;
        p.set(k, std::move(v));
        changed = true;
    };

    if (trained_ctx > 0) {
        fill("n_ctx", static_cast<int>(std::min<int64_t>(trained_ctx, MAX_CTX)));
    }

    auto sval = [&](const char * key) -> std::string {
        const std::string full = std::string("general.sampling.") + key;
        if (gguf_meta.is_object() && gguf_meta.contains(full) && gguf_meta[full].is_string()) {
            return gguf_meta[full].get<std::string>();
        }
        return "";
    };
    int   iv = 0;
    float fv = 0.0f;
    const std::string s_temp   = sval("temp");
    const std::string s_top_k  = sval("top_k");
    const std::string s_top_p  = sval("top_p");
    const std::string s_repeat = sval("repeat_penalty");
    if (!s_temp.empty()   && parse_float(s_temp, fv) && fv >= 0.0f && fv <= MAX_TEMP) fill("temp", fv);
    if (!s_top_k.empty()  && parse_int(s_top_k, iv)  && iv > 0 && iv <= MAX_TOP_K)    fill("top_k", iv);
    if (!s_top_p.empty()  && parse_float(s_top_p, fv) && fv > 0.0f && fv <= 1.0f)     fill("top_p", fv);
    if (!s_repeat.empty() && parse_float(s_repeat, fv) && fv >= 1.0f && fv <= 100.0f) fill("repeat_penalty", fv);

    fill("flash_attn", true);
    fill("type_k", std::string("turbo4"));
    fill("type_v", std::string("turbo3"));
    return changed;
}

static std::string now_iso() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    if (!std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t))) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(t));
    }
    return buf;
}

static std::string expand_home(const std::string & p) {
    if (p.size() > 1 && p[0] == '~' && (p[1] == '/' || p[1] == '\\')) {
        const char * home = std::getenv("HOME");
        if (home && home[0]) return std::string(home) + p.substr(1);
    }
    return p;
}

static std::string format_size(uint64_t bytes) {
    char buf[32];
    if (bytes >= 1073741824ULL)      std::snprintf(buf, sizeof(buf), "%.1f GB", static_cast<double>(bytes) / 1073741824.0);
    else if (bytes >= 1048576ULL)    std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(bytes) / 1048576.0);
    else if (bytes >= 1024ULL)       std::snprintf(buf, sizeof(buf), "%.1f KB", static_cast<double>(bytes) / 1024.0);
    else                             std::snprintf(buf, sizeof(buf), "%llu B", static_cast<unsigned long long>(bytes));
    return buf;
}

static std::string slugify(const std::string & s) {
    std::string out;
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (std::isalnum(c) || c == '.' || c == '-' || c == '_') {
            out += static_cast<char>(std::tolower(c));
        } else if (c == ':' || c == '/' || c == ' ' || c == '@') {
            out += '-';
        }
    }
    while (out.size() > 1 && out.back() == '-') out.pop_back();
    if (out.empty()) out = "model";
    return out;
}

inline std::string history_path()  { return config_dir() + "/history"; }
inline std::string presets_dir()  { return config_dir() + "/presets"; }

static std::vector<std::string> load_history_lines() {
    std::vector<std::string> out;
    std::ifstream f(history_path());
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

static void append_history_line(const std::string & s) {
    if (s.empty()) return;
    std::ofstream f(history_path(), std::ios::app);
    if (f) f << s << "\n";
}

static std::string load_preset_text(const std::string & name) {
    std::string n = name;
    if (n.size() > 1 && n[0] == '@') n = n.substr(1);
    for (const char * ext : {".txt", ".md", ""}) {
        const std::string p = presets_dir() + "/" + n + ext;
        std::ifstream f(p);
        if (f) return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    }
    return "";
}

static std::vector<std::string> list_presets() {
    std::vector<std::string> out;
    std::error_code ec;
    if (!std::filesystem::exists(presets_dir(), ec)) return out;
    for (const auto & de : std::filesystem::directory_iterator(presets_dir(), ec)) {
        if (!de.is_regular_file(ec)) continue;
        std::string n = de.path().filename().string();
        for (const char * ext : {".txt", ".md"}) {
            const size_t l = std::strlen(ext);
            if (n.size() > l && n.compare(n.size() - l, l, ext) == 0) {
                n = n.substr(0, n.size() - l);
                break;
            }
        }
        out.push_back(n);
    }
    return out;
}

static std::string session_dir_for(const std::string & model) {
    const std::string dir = sessions_dir() + "/" + slugify(model);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

static std::string new_session_path(const std::string & model) {
    const std::string dir = session_dir_for(model);
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    if (!std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t))) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(t));
    }
    return dir + "/" + buf + ".jsonl";
}

static std::string latest_session_for(const std::string & model) {
    const std::string dir = sessions_dir() + "/" + slugify(model);
    std::error_code ec;
    std::string best;
    std::filesystem::file_time_type best_t{};
    if (!std::filesystem::exists(dir, ec)) return "";
    for (const auto & de : std::filesystem::directory_iterator(dir, ec)) {
        if (!de.is_regular_file(ec) || de.path().extension() != ".jsonl") continue;
        const auto ft = de.last_write_time(ec);
        if (best.empty() || ft > best_t) { best = de.path().string(); best_t = ft; }
    }
    return best;
}

static bool save_session(const std::string & path, const std::vector<ChatMessage> & msgs) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return false;
    for (const auto & m : msgs) {
        nlohmann::json j;
        j["role"] = m.role;
        j["content"] = m.content;
        if (!m.image_path.empty()) j["image"] = m.image_path;
        if (!m.tool_calls_json.empty()) j["tool_calls"] = m.tool_calls_json;
        if (!m.tool_call_id.empty()) j["tool_call_id"] = m.tool_call_id;
        f << j.dump() << "\n";
    }
    f.flush();
    return f.good();
}

static bool load_session(const std::string & path, std::vector<ChatMessage> & msgs) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        try {
            const nlohmann::json j = nlohmann::json::parse(line);
            ChatMessage m;
            if (j.contains("role")) m.role = j["role"].get<std::string>();
            if (j.contains("content")) m.content = j["content"].get<std::string>();
            if (j.contains("image")) m.image_path = j["image"].get<std::string>();
            if (j.contains("tool_calls")) m.tool_calls_json = j["tool_calls"].get<std::string>();
            if (j.contains("tool_call_id")) m.tool_call_id = j["tool_call_id"].get<std::string>();
            msgs.push_back(std::move(m));
        } catch (const nlohmann::json::exception &) {}
    }
    return !msgs.empty();
}

bool save_models(const std::vector<ModelEntry> & models);

std::vector<ModelEntry> load_models() {
    std::vector<ModelEntry> models;
    std::ifstream f(models_json_path());
    if (!f) return models;
    nlohmann::json root;
    try {
        f >> root;
        if (root.contains("models") && root["models"].is_array()) {
            for (const auto & j : root["models"]) {
                try { models.push_back(ModelEntry::from_json(j)); }
                catch (const nlohmann::json::exception &) {  }
            }
        }
    } catch (const nlohmann::json::exception & e) {
        fprintf(stderr, "\033[33mwarning: %s is unreadable (%s); starting empty\033[0m\n",
                models_json_path().c_str(), e.what());
    }

    bool changed = false;
    for (auto & e : models) {
        if (e.seeded) continue;
        if (seed_model_profile(e.profile, e.trained_ctx, e.gguf_meta)) changed = true;
        e.seeded = true;
        changed = true;
    }
    if (changed) save_models(models);
    return models;
}

bool save_models(const std::vector<ModelEntry> & models) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(config_dir(), ec);
    if (ec) {
        fprintf(stderr, "\033[33mwarning: could not create config dir %s\033[0m\n", config_dir().c_str());
        return false;
    }
    nlohmann::json root;
    root["version"] = 1;
    root["models"] = nlohmann::json::array();
    for (const auto & e : models) root["models"].push_back(e.to_json());

    const std::string tmp = models_json_path() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) {
            fprintf(stderr, "\033[33mwarning: could not write %s\033[0m\n", tmp.c_str());
            return false;
        }
        f << root.dump(2) << "\n";
        f.flush();
        if (!f) {
            fprintf(stderr, "\033[33mwarning: failed writing %s\033[0m\n", tmp.c_str());
            return false;
        }
    }
    fs::rename(tmp, models_json_path(), ec);
    if (ec) {
        fprintf(stderr, "\033[33mwarning: could not rename %s into place: %s\033[0m\n",
                models_json_path().c_str(), ec.message().c_str());
        return false;
    }
    return true;
}

ModelEntry * find_model(std::vector<ModelEntry> & models, const std::string & name) {
    for (auto & m : models) if (m.name == name) return &m;
    return nullptr;
}
const ModelEntry * find_model(const std::vector<ModelEntry> & models, const std::string & name) {
    for (const auto & m : models) if (m.name == name) return &m;
    return nullptr;
}

struct ModelMeta {
    std::string desc;
    int64_t     trained_ctx = 0;
    nlohmann::json gguf_meta;
};

static ModelMeta read_model_meta(const std::string & path) {
    ModelMeta meta;
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.vocab_only = true;
    LlamaModel m(llama_model_load_from_file(path.c_str(), mparams));
    if (!m) return meta;

    nlohmann::json meta_map = nlohmann::json::object();
    const int32_t count = llama_model_meta_count(m);
    if (count > 0 && count <= 4096) {
        for (int32_t i = 0; i < count; i++) {
            const int32_t klen = llama_model_meta_key_by_index(m, i, nullptr, 0);
            if (klen <= 0 || klen > (1 << 20)) continue;
            std::string key(static_cast<size_t>(klen) + 1, '\0');
            if (llama_model_meta_key_by_index(m, i, key.data(), key.size()) < 0) continue;
            key.resize(static_cast<size_t>(klen));

            const int32_t vlen = llama_model_meta_val_str_by_index(m, i, nullptr, 0);
            if (vlen < 0 || vlen > (1 << 20)) continue;
            std::string val(static_cast<size_t>(vlen) + 1, '\0');
            if (llama_model_meta_val_str_by_index(m, i, val.data(), val.size()) < 0) continue;
            val.resize(static_cast<size_t>(vlen));
            meta_map[key] = val;
        }
    }

    auto meta_str = [&](const std::string & key) -> std::string {
        if (meta_map.contains(key) && meta_map[key].is_string()) {
            return meta_map[key].get<std::string>();
        }
        return "";
    };
    const std::string arch = meta_str("general.architecture");
    std::string name = meta_str("general.name");
    if (name.empty()) name = meta_str("general.basename");

    std::string ctx = meta_str("general.context_length");
    if (ctx.empty() && !arch.empty()) ctx = meta_str(arch + ".context_length");
    if (!ctx.empty()) {

        size_t i = 0;
        while (i < ctx.size() && !std::isdigit(static_cast<unsigned char>(ctx[i]))) i++;
        if (i < ctx.size()) {
            errno = 0;
            char * end = nullptr;
            const long long v = std::strtoll(ctx.c_str() + i, &end, 10);
            if (errno != ERANGE && end != ctx.c_str() + i && v > 0) meta.trained_ctx = v;
        }
    }

    if (!name.empty()) meta.desc = arch.empty() ? name : name + " (arch: " + arch + ")";
    else if (!arch.empty()) meta.desc = arch;

    meta.gguf_meta = std::move(meta_map);
    return meta;
}

static bool valid_kv_name(const std::string & n) {
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) {
        if (n == KV_OPTIONS[i].short_name) return true;
    }
    return false;
}

static bool parse_profile_value(const std::string & key, const std::string & val, ModelProfile & p) {
    int iv = 0; float fv = 0.0f;
    if (key == "n_ctx")      { if (!parse_int(val, iv) || iv < 1 || iv > MAX_CTX) return false; p.set(key, iv); return true; }
    if (key == "ngl")        { if (!parse_int(val, iv) || iv < -1 || iv > 10000) return false; p.set(key, iv); return true; }
    if (key == "n_threads")  { if (!parse_int(val, iv) || iv < 1 || iv > MAX_THREADS) return false; p.set(key, iv); return true; }
    if (key == "top_k")      { if (!parse_int(val, iv) || iv < 0 || iv > MAX_TOP_K) return false; p.set(key, iv); return true; }
    if (key == "temp")       { if (!parse_float(val, fv) || fv < 0.0f || fv > MAX_TEMP) return false; p.set(key, fv); return true; }
    if (key == "top_p")      { if (!parse_float(val, fv) || fv < 0.0f || fv > 1.0f) return false; p.set(key, fv); return true; }
    if (key == "repeat_penalty") { if (!parse_float(val, fv) || fv < 1.0f || fv > 100.0f) return false; p.set(key, fv); return true; }
    if (key == "flash_attn" || key == "mtp") {
        if (val != "true" && val != "false" && val != "1" && val != "0" && val != "on" && val != "off") return false;
        p.set(key, (val == "true" || val == "1" || val == "on")); return true;
    }
    if (key == "type_k" || key == "type_v") {
        if (!valid_kv_name(val)) return false;
        p.set(key, val); return true;
    }
    if (key == "system_prompt") { p.set(key, val); return true; }
    return false;
}

static bool looks_like_file(const std::string & arg) {
    if (arg.find('/') != std::string::npos || arg.find('\\') != std::string::npos) return true;
    if (arg.size() > 5 && arg.compare(arg.size() - 5, 5, ".gguf") == 0) return true;
    if (std::filesystem::exists(expand_home(arg))) return true;
    return false;
}

bool resolve_model_arg(const std::string & arg, std::string & path_out, std::string & friendly_out) {
    const std::vector<ModelEntry> models = load_models();
    if (const ModelEntry * e = find_model(models, arg)) {
        path_out = e->path;
        friendly_out = e->name;
        return true;
    }
    if (looks_like_file(arg)) {
        path_out = expand_home(arg);
        friendly_out.clear();
        return true;
    }
    return false;
}

ModelEntry * auto_register(std::vector<ModelEntry> & models,
                           const std::string & path, const std::string & friendly,
                           const ModelMeta & meta) {
    if (ModelEntry * existing = find_model(models, friendly)) return existing;
    std::error_code ec;
    const std::filesystem::path canon = std::filesystem::canonical(path, ec);
    if (!ec) {
        for (auto & m : models) {
            std::error_code ec2;
            const std::filesystem::path mcanon = std::filesystem::canonical(m.path, ec2);
            if (!ec2 && mcanon == canon) return &m;
        }
    }
    ModelEntry e;
    e.name = friendly;
    e.path = path;
    e.source = "local";
    e.added = now_iso();
    const auto sz = std::filesystem::file_size(path, ec);
    e.size_bytes = ec ? 0 : sz;
    e.desc = meta.desc;
    e.trained_ctx = meta.trained_ctx;
    e.gguf_meta = meta.gguf_meta;
    seed_model_profile(e.profile, meta.trained_ctx, meta.gguf_meta);
    e.seeded = true;
    models.push_back(std::move(e));
    return &models.back();
}

int cmd_rm(const std::vector<std::string> & args) {
    if (args.empty()) {
        fprintf(stderr, "usage: anvil rm <name> [--yes]\n");
        return 1;
    }
    const std::string name = args[0];
    bool delete_file = false;
    for (size_t i = 1; i < args.size(); i++) {
        if (args[i] == "--yes") delete_file = true;
        else {
            fprintf(stderr, "error: unknown option '%s'\n", args[i].c_str());
            return 1;
        }
    }
    std::vector<ModelEntry> models = load_models();
    auto it = std::find_if(models.begin(), models.end(),
                           [&](const ModelEntry & e) { return e.name == name; });
    if (it == models.end()) {
        fprintf(stderr, "error: no model named '%s'\n", name.c_str());
        return 1;
    }
    const std::string path = it->path;
    if (delete_file) {
        std::error_code ec;
        if (!std::filesystem::remove(path, ec)) {
            fprintf(stderr, "warning: could not delete %s (%s)\n", path.c_str(), ec.message().c_str());
        } else {
            fprintf(stderr, "Deleted %s\n", path.c_str());
        }
    }
    models.erase(it);
    if (!save_models(models)) return 1;
    printf("Removed '%s' from the registry.\n", name.c_str());
    return 0;
}

int cmd_models(const std::vector<std::string> & args) {

    if (args.size() > 1 && args[0] == "list") {
        fprintf(stderr, "error: unexpected argument '%s'\n", args[1].c_str());
        return 1;
    }
    if (args.empty() || args[0] == "list") {
        const std::vector<ModelEntry> models = load_models();
        if (models.empty()) {
            printf("No models registered.\n");
            printf("  anvil pull ollama:<name>[:tag]      pull from the Ollama registry\n");
            printf("  anvil pull hf:<repo>[:file]         pull from HuggingFace\n");
            printf("  anvil models import <file.gguf> [--name <name>]\n");
            return 0;
        }
        printf("%-26s %-12s %-10s %-10s %s\n", "NAME", "SOURCE", "SIZE", "TRAINED", "PATH");
        for (const auto & m : models) {
            char trained[32];
            if (m.trained_ctx > 0) std::snprintf(trained, sizeof(trained), "%lld", static_cast<long long>(m.trained_ctx));
            else std::snprintf(trained, sizeof(trained), "-");
            printf("%-26s %-12s %-10s %-10s %s\n", m.name.c_str(), m.source.c_str(),
                   format_size(m.size_bytes).c_str(), trained, m.path.c_str());
        }
        return 0;
    }

    if (args[0] == "--help" || args[0] == "-h") {
        printf("usage: anvil models [list] | import <file.gguf> [--name <name>] | rm <name> [--yes]\n");
        return 0;
    }

    if (args[0] == "import") {
        if (args.size() < 2) {
            fprintf(stderr, "usage: anvil models import <file.gguf> [--name <name>]\n");
            return 1;
        }
        std::string path = expand_home(args[1]);
        std::string name;
        for (size_t i = 2; i < args.size(); i++) {
            if (args[i] == "--name" && i + 1 < args.size()) name = args[++i];
            else {
                fprintf(stderr, "error: unknown import option '%s'\n", args[i].c_str());
                return 1;
            }
        }
        if (!validate_gguf(path)) {
            fprintf(stderr, "\033[31merror: %s\033[0m\n", gguf_check_error(path).c_str());
            return 1;
        }
        std::error_code ec;
        const std::filesystem::path canon = std::filesystem::canonical(path, ec);
        if (!ec) path = canon.string();

        std::vector<ModelEntry> models = load_models();
        if (name.empty()) name = slugify(std::filesystem::path(path).stem().string());
        if (find_model(models, name)) {
            fprintf(stderr, "error: a model named '%s' is already registered\n", name.c_str());
            return 1;
        }
        ModelEntry e;
        e.name = name;
        e.path = path;
        e.source = "local";
        e.added = now_iso();
        e.size_bytes = std::filesystem::file_size(path, ec);
        if (ec) e.size_bytes = 0;
        const ModelMeta meta = read_model_meta(path);
        e.desc = meta.desc;
        e.trained_ctx = meta.trained_ctx;
        e.gguf_meta = meta.gguf_meta;
        seed_model_profile(e.profile, meta.trained_ctx, meta.gguf_meta);
        e.seeded = true;
        models.push_back(std::move(e));
        if (!save_models(models)) return 1;
        printf("Registered '%s' -> %s\n", name.c_str(), path.c_str());
        printf("  run with: anvil run %s\n", name.c_str());
        return 0;
    }

    if (args[0] == "rm" || args[0] == "remove") {
        return cmd_rm(std::vector<std::string>(args.begin() + 1, args.end()));
    }

    fprintf(stderr, "unknown 'models' subcommand '%s' (try: list, import, rm)\n", args[0].c_str());
    return 1;
}

int cmd_profile(const std::vector<std::string> & args) {
    if (args.empty()) {
        fprintf(stderr, "usage: anvil profile <name> [set k=v ...] [unset k ...] [reset]\n");
        return 1;
    }
    const std::string name = args[0];
    std::vector<ModelEntry> models = load_models();
    ModelEntry * e = find_model(models, name);
    if (!e) {
        fprintf(stderr, "error: no model named '%s'\n", name.c_str());
        return 1;
    }

    auto is_cmd = [](const std::string & s) { return s == "set" || s == "unset" || s == "reset"; };
    bool changed = false;
    bool show = true;
    for (size_t i = 1; i < args.size(); i++) {
        const std::string & a = args[i];
        if (a == "set") {
            show = false;
            const size_t before = i;
            while (i + 1 < args.size() && !is_cmd(args[i + 1])) {
                const std::string kv = args[++i];
                const size_t eq = kv.find('=');
                if (eq == std::string::npos) {
                    fprintf(stderr, "error: expected key=value, got '%s'\n", kv.c_str());
                    return 1;
                }
                const std::string key = kv.substr(0, eq);
                const std::string val = kv.substr(eq + 1);
                if (!ModelProfile::valid_key(key)) {
                    fprintf(stderr, "error: unknown profile key '%s' (valid: n_ctx ngl n_threads temp top_k top_p repeat_penalty flash_attn mtp type_k type_v system_prompt)\n", key.c_str());
                    return 1;
                }
                if (!parse_profile_value(key, val, e->profile)) {
                    fprintf(stderr, "error: invalid value '%s' for '%s'\n", val.c_str(), key.c_str());
                    return 1;
                }
                changed = true;
            }
            if (i == before) {
                fprintf(stderr, "usage: anvil profile %s set key=value ...\n", name.c_str());
                return 1;
            }
        } else if (a == "unset") {
            show = false;
            const size_t before = i;
            while (i + 1 < args.size() && !is_cmd(args[i + 1])) {
                const std::string key = args[++i];
                if (!ModelProfile::valid_key(key)) {
                    fprintf(stderr, "error: unknown profile key '%s'\n", key.c_str());
                    return 1;
                }
                if (e->profile.unset(key)) changed = true;
            }
            if (i == before) {
                fprintf(stderr, "usage: anvil profile %s unset key ...\n", name.c_str());
                return 1;
            }
        } else if (a == "reset") {
            show = false;
            e->profile.clear();
            changed = true;
        } else {
            fprintf(stderr, "error: unknown profile command '%s' (try: set, unset, reset)\n", a.c_str());
            return 1;
        }
    }

    if (show) {
        printf("Profile: %s  (path: %s)\n", e->name.c_str(), e->path.c_str());
        if (e->profile.empty()) {
            printf("  (empty — inherits global config defaults)\n");
        } else {
            for (const auto & [k, v] : e->profile.settings) {
                std::string s;
                if (v.is_string())            s = v.get<std::string>();
                else if (v.is_number_float()) { char b[32]; std::snprintf(b, sizeof(b), "%.4g", v.get<double>()); s = b; }
                else if (v.is_boolean())       s = v.get<bool>() ? "true" : "false";
                else                          s = v.dump();
                printf("  %-18s = %s\n", k.c_str(), s.c_str());
            }
        }
        return 0;
    }
    if (changed) {
        if (!save_models(models)) return 1;
        printf("Profile updated for '%s'.\n", e->name.c_str());
    }
    return 0;
}

int cmd_pull(const std::vector<std::string> & args);

namespace {

constexpr const char * OLLAMA_BASE = "https://registry.ollama.ai/v2";

bool valid_registry_component(const std::string & s) {
    if (s.empty() || s.size() > 128) return false;
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (!(std::isalnum(c) || c == '.' || c == '-' || c == '_')) return false;
    }
    return true;
}

bool valid_digest(const std::string & s) {
    if (s.compare(0, 7, "sha256:") != 0) return false;
    const std::string hex = s.substr(7);
    if (hex.size() != 64) return false;
    for (const char ch : hex) {
        if (!std::isxdigit(static_cast<unsigned char>(ch))) return false;
    }
    return true;
}

bool valid_hf_sha(const std::string & s) {
    if (s.size() != 40) return false;
    for (const char ch : s) {
        if (!std::isxdigit(static_cast<unsigned char>(ch))) return false;
    }
    return true;
}

bool valid_hf_filename(const std::string & s) {
    if (s.empty() || s.size() > 256) return false;
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (!(std::isalnum(c) || c == '.' || c == '-' || c == '_')) return false;
    }
    return true;
}

std::string pull_tmp_path() {
    static std::atomic<unsigned> counter{0};
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = config_dir();
    return (dir / ("anvil-cmd-" + std::to_string(counter.fetch_add(1)) +
                   "-" + std::to_string(std::chrono::steady_clock::now()
                                            .time_since_epoch().count()) + ".tmp")).string();
}

std::string capture(const std::string & cmd) {
    const std::string tmp = pull_tmp_path();
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    const std::string full = "{ " + cmd + "; } > \"" + tmp + "\" 2>/dev/null";
    std::string out;
    if (system(full.c_str()) == 0) {
        std::ifstream f(tmp, std::ios::binary);
        out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    std::filesystem::remove(tmp, ec);
    return out;
}

inline std::string g_hf_token;
static volatile sig_atomic_t g_serve_stop = 0;

static std::string shell_quote(const std::string & s) {
    std::string out = "'";
    for (const char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static std::string auth_flag() {
    if (g_hf_token.empty()) return "";
    return " -H " + shell_quote("Authorization: Bearer " + g_hf_token);
}

std::string http_get(const std::string & url, const std::string & extra_flags = "") {
    std::string out = capture("curl -fsSL --max-time 60 " + extra_flags + auth_flag() + " " + shell_quote(url));
    if (out.empty()) {
        out = capture("wget -qO- --timeout=60 --header=" + shell_quote("Authorization: Bearer " + g_hf_token) + " " + shell_quote(url));
    }
    return out;
}

int http_download(const std::string & url, const std::string & out_path,
                  uint64_t expected_size = 0) {
    const std::string part = out_path + ".part";
    if (expected_size > 0) {
        std::error_code st;
        const uint64_t cur = std::filesystem::file_size(part, st);
        if (!st && cur > expected_size) {
            fprintf(stderr, "  stale partial (%llu B > expected %llu B); restarting\n",
                    static_cast<unsigned long long>(cur),
                    static_cast<unsigned long long>(expected_size));
            std::filesystem::remove(part, st);
        }
    }

    std::string cmd = std::string("curl --fail --location --progress-bar --retry 3 --retry-delay 2 ") +
                      "--connect-timeout 20 --continue-at - " + auth_flag() + " -o " +
                      shell_quote(part) + " " + shell_quote(url);
    const int rc = system(cmd.c_str());
    if (rc != 0) {
        fprintf(stderr, "\033[31mpull failed (curl exit %d). Partial download kept at %s — retry to resume.\033[0m\n",
                rc, part.c_str());
        return -1;
    }
    std::error_code ec;
    std::filesystem::rename(part, out_path, ec);
    if (ec) {
        fprintf(stderr, "\033[31merror: could not finalize %s (%s)\033[0m\n",
                out_path.c_str(), ec.message().c_str());
        return -1;
    }
    return 0;
}

std::string sha256_of(const std::string & path) {
    std::string hex = capture("sha256sum " + shell_quote(path) + " | awk '{print $1}'");
    while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) hex.pop_back();
    if (!hex.empty()) return hex;
    hex = capture("shasum -a 256 " + shell_quote(path) + " | awk '{print $1}'");
    while (!hex.empty() && (hex.back() == '\n' || hex.back() == '\r')) hex.pop_back();
    return hex;
}

struct OllamaLayer {
    std::string media_type;
    std::string digest;
    uint64_t    size = 0;
};

struct OllamaManifest {
    std::vector<OllamaLayer> layers;
    uint64_t total_size = 0;
};

bool parse_ollama_manifest(const std::string & json, OllamaManifest & out) {
    try {
        const nlohmann::json j = nlohmann::json::parse(json);
        if (!j.contains("layers") || !j["layers"].is_array()) return false;
        for (const auto & l : j["layers"]) {
            OllamaLayer layer;
            if (l.contains("mediaType")) layer.media_type = l["mediaType"].get<std::string>();
            if (l.contains("digest"))    layer.digest    = l["digest"].get<std::string>();
            if (l.contains("size"))      layer.size      = l["size"].get<uint64_t>();

            std::string digest = layer.digest;
            for (char & c : digest) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            if (!valid_digest(digest)) continue;
            layer.digest = std::move(digest);
            out.layers.push_back(std::move(layer));
        }
        return !out.layers.empty();
    } catch (const nlohmann::json::exception &) {
        return false;
    }
}

void apply_ollama_params(const nlohmann::json & params, ModelProfile & p) {
    if (!params.is_object()) return;
    auto num = [&](const char * k, float & out) -> bool {
        if (!params.contains(k) || !params[k].is_number()) return false;
        out = params[k].get<float>();
        return true;
    };
    float f = 0.0f;
    if (num("temperature", f) && f >= 0.0f && f <= MAX_TEMP)     p.set("temp", f);
    if (num("top_p", f) && f >= 0.0f && f <= 1.0f)               p.set("top_p", f);
    if (num("repeat_penalty", f) && f >= 1.0f && f <= 100.0f)    p.set("repeat_penalty", f);
    if (num("top_k", f)) {
        const int iv = static_cast<int>(f);
        if (iv >= 0 && iv <= MAX_TOP_K) p.set("top_k", iv);
    }
    if (num("num_ctx", f)) {
        const int iv = static_cast<int>(f);
        if (iv >= 1 && iv <= MAX_CTX) p.set("n_ctx", iv);
    }
}

bool parse_ollama_spec(const std::string & spec,
                       std::string & ns, std::string & name, std::string & tag) {
    ns = "library";
    name = spec;
    tag = "latest";
    const size_t slash = spec.find('/');
    if (slash != std::string::npos) {
        ns = spec.substr(0, slash);
        name = spec.substr(slash + 1);
    }
    const size_t colon = name.find(':');
    if (colon != std::string::npos) {
        tag = name.substr(colon + 1);
        name = name.substr(0, colon);
    }
    return valid_registry_component(ns) && valid_registry_component(name) &&
           valid_registry_component(tag);
}

int register_pulled(const std::string & friendly, const std::string & path,
                    const std::string & source, const std::string & source_id,
                    const std::string & template_text, const std::string & license,
                    const std::string & params_json) {
    std::vector<ModelEntry> models = load_models();
    if (find_model(models, friendly)) {
        fprintf(stderr, "\033[31merror: a model named '%s' is already registered (anvil rm %s to replace)\033[0m\n",
                friendly.c_str(), friendly.c_str());
        return 1;
    }
    const ModelMeta meta = read_model_meta(path);
    if (meta.desc.empty() && meta.trained_ctx <= 0) {
        fprintf(stderr, "\033[33mwarning: could not read GGUF metadata from %s (file may be incomplete)\033[0m\n",
                path.c_str());
    }
    ModelEntry e;
    e.name = friendly;
    e.path = path;
    e.source = source;
    e.source_id = source_id;
    e.added = now_iso();
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    e.size_bytes = ec ? 0 : sz;
    e.desc = meta.desc;
    e.trained_ctx = meta.trained_ctx;
    e.gguf_meta = meta.gguf_meta;
    e.chat_template = template_text;
    e.license = license;
    e.params_json = params_json;

    try {
        if (!params_json.empty()) apply_ollama_params(nlohmann::json::parse(params_json), e.profile);
    } catch (const nlohmann::json::exception &) {}

    {
        auto it = e.profile.settings.find("n_ctx");
        if (it != e.profile.settings.end() && it->second.is_number_integer()) {
            const int64_t nctx = it->second.get<int64_t>();
            if (nctx > MAX_CTX) it->second = static_cast<int64_t>(MAX_CTX);
        }
    }
    seed_model_profile(e.profile, e.trained_ctx, e.gguf_meta);
    e.seeded = true;
    models.push_back(std::move(e));
    if (!save_models(models)) return 1;
    printf("\033[32mRegistered '%s' (%s) -> %s\033[0m\n", friendly.c_str(),
           format_size(e.size_bytes).c_str(), path.c_str());
    if (!e.desc.empty())      printf("  arch       : %s\n", e.desc.c_str());
    if (e.trained_ctx > 0)    printf("  trained ctx: %lld\n", static_cast<long long>(e.trained_ctx));
    if (!e.chat_template.empty()) printf("  template   : %zu bytes (from ollama metadata)\n", e.chat_template.size());
    if (!e.profile.empty()) {
        printf("  profile    : ");
        bool first = true;
        for (const auto & [k, v] : e.profile.settings) {
            std::string s;
            if (v.is_string())            s = v.get<std::string>();
            else if (v.is_number_float()) { char b[32]; std::snprintf(b, sizeof(b), "%.4g", v.get<double>()); s = b; }
            else if (v.is_boolean())       s = v.get<bool>() ? "true" : "false";
            else                          s = v.dump();
            printf("%s%s=%s", first ? "" : " ", k.c_str(), s.c_str());
            first = false;
        }
        printf(" (from ollama params)\n");
    }
    printf("  run        : anvil run %s\n", friendly.c_str());
    return 0;
}

bool verify_digest(const std::string & path, const std::string & digest) {
    if (digest.compare(0, 7, "sha256:") != 0) return true;
    const std::string expected = digest.substr(7);
    const std::string actual = sha256_of(path);
    if (actual.empty()) {
        fprintf(stderr, "\033[33mwarning: no sha256 tool available; skipping verification\033[0m\n");
        return true;
    }
    if (actual != expected) {
        fprintf(stderr, "\033[31mchecksum mismatch for %s (got %s, expected %s)\033[0m\n",
                path.c_str(), actual.c_str(), expected.c_str());
        return false;
    }
    printf("  verified sha256:%s\n", expected.c_str());
    return true;
}

int download_verified(const std::string & url, const std::string & dest,
                      uint64_t expected_size, const std::string & digest) {
    for (int attempt = 0; attempt < 2; attempt++) {
        if (http_download(url, dest, expected_size) != 0) return -1;
        if (verify_digest(dest, digest)) return 0;
        std::error_code ec;
        std::filesystem::remove(dest, ec);
        std::filesystem::remove(dest + ".part", ec);
        if (attempt == 0) fprintf(stderr, "  checksum mismatch; retrying from scratch...\n");
    }
    fprintf(stderr, "\033[31merror: checksum still mismatched after retry; giving up\033[0m\n");
    return -1;
}

int pull_blob(const std::string & ns, const std::string & model, const OllamaLayer & layer,
              const std::string & dest) {
    const std::string url = std::string(OLLAMA_BASE) + "/" + ns + "/" + model + "/blobs/" + layer.digest;
    printf("  %-28s %s\n", layer.media_type.c_str(), format_size(layer.size).c_str());
    return download_verified(url, dest, layer.size, layer.digest);
}

struct HfFile {
    std::string path;
    uint64_t    size = 0;
    std::string oid;
};

bool parse_hf_spec(const std::string & spec, std::string & repo, std::string & file) {
    repo = spec;
    file.clear();
    const size_t colon = spec.find(':');
    if (colon != std::string::npos) {
        repo = spec.substr(0, colon);
        file = spec.substr(colon + 1);
    }
    const size_t slash = repo.find('/');
    if (repo.empty() || slash == std::string::npos || slash == 0 || slash + 1 >= repo.size()) return false;
    for (const char ch : repo) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (!(std::isalnum(c) || c == '/' || c == '.' || c == '-' || c == '_')) return false;
    }

    if (repo.find("/.") != std::string::npos || repo.find("./") != std::string::npos ||
        repo.find("..") != std::string::npos) return false;
    if (file.empty()) return true;
    if (file.size() > 256 || file.find('/') != std::string::npos ||
        file.find("..") != std::string::npos) return false;
    for (const char ch : file) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (!(std::isalnum(c) || c == '.' || c == '-' || c == '_')) return false;
    }
    return true;
}

bool resolve_hf_sha(const std::string & repo, std::string & sha) {
    const std::string url = "https://huggingface.co/api/models/" + repo;
    const std::string body = http_get(url);
    if (body.empty()) return false;
    try {
        const nlohmann::json j = nlohmann::json::parse(body);
        if (j.contains("sha") && j["sha"].is_string()) {
            const std::string s = j["sha"].get<std::string>();
            if (valid_hf_sha(s)) { sha = s; return true; }
        }
    } catch (const nlohmann::json::exception &) {}
    return false;
}

bool list_hf_files(const std::string & repo, const std::string & sha,
                   std::vector<HfFile> & out) {
    const std::string url = "https://huggingface.co/api/models/" + repo + "/tree/" + sha + "?recursive=true";
    const std::string body = http_get(url);
    if (body.empty()) return false;
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(body);
    } catch (const nlohmann::json::exception &) {
        return false;
    }
    if (!j.is_array()) return false;
    for (const auto & f : j) {

        if (!f.is_object() || !f.contains("type") || !f["type"].is_string()) continue;
        std::string type;
        try { type = f["type"].get<std::string>(); } catch (const nlohmann::json::exception &) { continue; }
        if (type != "file") continue;
        if (!f.contains("path") || !f["path"].is_string()) continue;
        std::string p;
        try { p = f["path"].get<std::string>(); } catch (const nlohmann::json::exception &) { continue; }
        if (p.size() < 5 || p.compare(p.size() - 5, 5, ".gguf") != 0) continue;
        if (p.find('/') != std::string::npos) continue;
        if (!valid_hf_filename(p)) continue;
        HfFile hf;
        hf.path = p;
        if (f.contains("size") && f["size"].is_number()) {
            try { hf.size = f["size"].get<uint64_t>(); } catch (const nlohmann::json::exception &) {}
        }
        if (f.contains("lfs") && f["lfs"].is_object() && f["lfs"].contains("oid") &&
            f["lfs"]["oid"].is_string()) {
            try { hf.oid = f["lfs"]["oid"].get<std::string>(); } catch (const nlohmann::json::exception &) {}
        }
        out.push_back(std::move(hf));
    }

    std::sort(out.begin(), out.end(),
              [](const HfFile & a, const HfFile & b) { return a.size < b.size; });
    return true;
}

int pick_hf_file(const std::vector<HfFile> & files) {
    printf("%zu .gguf file(s):\n", files.size());
    for (size_t i = 0; i < files.size(); i++) {
        printf("  [%2zu] %-44s %s\n", i + 1, files[i].path.c_str(),
               format_size(files[i].size).c_str());
    }
    printf("Select a quant [1-%zu]: ", files.size());
    std::fflush(stdout);
    std::string line;
    if (!std::getline(std::cin, line)) return -1;
    int idx = 0;
    if (!parse_int(line, idx) || idx < 1 || idx > static_cast<int>(files.size())) return -1;
    return idx - 1;
}

}

int cmd_pull_hf(const std::string & spec, const std::vector<std::string> & extra) {
    bool list_only = false;
    for (const auto & a : extra) {
        if (a == "--list") list_only = true;
        else {
            fprintf(stderr, "error: unknown pull option '%s'\n", a.c_str());
            return 1;
        }
    }

    std::string repo, file;
    if (!parse_hf_spec(spec, repo, file)) {
        fprintf(stderr, "error: invalid HF spec '%s' (expected hf:<owner>/<repo>[:<file.gguf>])\n",
                spec.c_str());
        return 1;
    }

    if (file.empty() && !list_only && !stdin_is_tty()) {
        fprintf(stderr, "error: interactive picker needs a terminal; pass the file explicitly:\n"
                        "  anvil pull hf:%s:<file.gguf>\n  anvil pull hf:%s --list\n",
                repo.c_str(), repo.c_str());
        return 1;
    }

    std::string sha;
    if (!resolve_hf_sha(repo, sha)) {
        fprintf(stderr, "\033[31merror: could not access '%s' (repo not found, gated, or network issue)\033[0m\n",
                repo.c_str());
        return 1;
    }

    std::vector<HfFile> files;
    if (!list_hf_files(repo, sha, files)) {
        fprintf(stderr, "\033[31merror: could not list '%s' (repo not found, gated, or network issue)\033[0m\n",
                repo.c_str());
        return 1;
    }

    if (!file.empty() && list_only) {
        fprintf(stderr, "error: --list cannot be combined with an explicit file\n");
        return 1;
    }
    if (list_only) {
        if (files.empty()) {
            printf("No .gguf files in %s.\n", repo.c_str());
            return 0;
        }
        printf("%s (%zu .gguf):\n", repo.c_str(), files.size());
        for (const auto & f : files) {
            std::string suffix = f.oid.empty() ? "" : ("  sha256: " + f.oid);
            printf("  %-44s %s%s\n", f.path.c_str(), format_size(f.size).c_str(), suffix.c_str());
        }
        return 0;
    }

    if (files.empty()) {
        fprintf(stderr, "\033[31merror: no .gguf files in %s (safetensors-only repos need the HF converter: "
                        "backends/llama-turbo/convert_hf_to_gguf.py)\033[0m\n", repo.c_str());
        return 1;
    }

    const HfFile * chosen = nullptr;
    if (file.empty()) {
        const int idx = pick_hf_file(files);
        if (idx < 0) {
            fprintf(stderr, "error: invalid selection\n");
            return 1;
        }
        chosen = &files[static_cast<size_t>(idx)];
    } else {
        for (const auto & f : files) {
            if (f.path == file) { chosen = &f; break; }
        }
        if (!chosen) {
            fprintf(stderr, "\033[31merror: '%s' is not a .gguf file in %s (run 'anvil pull hf:%s --list')\033[0m\n",
                    file.c_str(), repo.c_str(), repo.c_str());
            return 1;
        }
    }

    const std::string friendly = slugify(std::filesystem::path(chosen->path).stem().string());
    const std::string source_id = repo + ":" + chosen->path;

    std::error_code ec;
    std::filesystem::create_directories(models_dir(), ec);
    if (ec) {
        fprintf(stderr, "\033[31merror: could not create %s\033[0m\n", models_dir().c_str());
        return 1;
    }
    const std::string model_path = models_dir() + "/" + friendly + ".gguf";
    if (std::filesystem::exists(model_path)) {
        fprintf(stderr, "\033[31merror: model file already exists at %s (anvil rm %s --yes to replace)\033[0m\n",
                model_path.c_str(), friendly.c_str());
        return 1;
    }

    const std::string url = "https://huggingface.co/" + repo + "/resolve/" + sha + "/" + chosen->path;
    printf("Downloading %s (%s)...\n", chosen->path.c_str(), format_size(chosen->size).c_str());

    std::string digest;
    if (chosen->oid.empty()) {
        fprintf(stderr, "\033[33mwarning: no sha256 available from the HF API; download will not be verified\033[0m\n");
    } else if (valid_digest("sha256:" + chosen->oid)) {
        digest = "sha256:" + chosen->oid;
    } else {
        fprintf(stderr, "\033[33mwarning: malformed LFS oid for %s; download will not be verified\033[0m\n",
                chosen->path.c_str());
    }
    if (download_verified(url, model_path, chosen->size, digest) != 0) return 1;

    return register_pulled(friendly, model_path, "hf", source_id, "", "", "");
}

int cmd_pull(const std::vector<std::string> & args) {
    if (args.empty()) {
        fprintf(stderr, "usage: anvil pull ollama:<name>[:tag] | ollama-local:<name>[:tag] | hf:<repo>[:file]\n");
        return 1;
    }

    std::string source = "ollama";
    std::string rest = args[0];
    const size_t colon = rest.find(':');
    if (colon != std::string::npos) {
        const std::string prefix = rest.substr(0, colon);
        if (prefix == "ollama" || prefix == "ollama-local" || prefix == "hf") {
            source = prefix;
            rest = rest.substr(colon + 1);
        }
    }
    if (source == "hf") {
        return cmd_pull_hf(rest, std::vector<std::string>(args.begin() + 1, args.end()));
    }

    if (args.size() > 1) {
        fprintf(stderr, "error: unexpected argument '%s'\n", args[1].c_str());
        return 1;
    }

    std::string ns, name, tag;
    if (!parse_ollama_spec(rest, ns, name, tag)) {
        fprintf(stderr, "error: invalid model spec '%s' (expected [ns/]name[:tag])\n", rest.c_str());
        return 1;
    }
    const std::string friendly = slugify(name + "-" + tag);
    const std::string source_id = ns + "/" + name + ":" + tag;

    OllamaManifest manifest;
    if (source == "ollama") {
        const std::string url = std::string(OLLAMA_BASE) + "/" + ns + "/" + name + "/manifests/" + tag;
        printf("Pulling from Ollama registry: %s\n", source_id.c_str());
        const std::string body = http_get(url, "-H \"Accept: application/vnd.docker.distribution.manifest.v2+json\"");
        if (body.empty() || !parse_ollama_manifest(body, manifest)) {
            fprintf(stderr, "\033[31merror: model '%s' not found in the Ollama registry (or network issue)\033[0m\n",
                    source_id.c_str());
            return 1;
        }
    } else {

        std::string base;
        const char * om = std::getenv("OLLAMA_MODELS");
        if (om && om[0]) {
            base = om;
            while (base.size() > 1 && base.back() == '/') base.pop_back();
        } else {
            const char * home = std::getenv("HOME");
#ifdef _WIN32
            if (!home || !home[0]) home = std::getenv("USERPROFILE");
#endif
            if (!home || !home[0]) home = ".";
            base = std::string(home) + "/.ollama/models";
        }
        const std::string mp = base + "/manifests/registry.ollama.ai/" + ns + "/" + name + "/" + tag;
        printf("Importing from local ollama install: %s\n", source_id.c_str());
        std::ifstream mf(mp);
        if (!mf) {
            fprintf(stderr, "\033[31merror: no local ollama model at %s (did you 'ollama pull %s' first?)\033[0m\n",
                    mp.c_str(), source_id.c_str());
            return 1;
        }
        const std::string body((std::istreambuf_iterator<char>(mf)), std::istreambuf_iterator<char>());
        if (!parse_ollama_manifest(body, manifest)) {
            fprintf(stderr, "\033[31merror: unreadable manifest at %s\033[0m\n", mp.c_str());
            return 1;
        }

        for (auto & layer : manifest.layers) {
            std::string fname = layer.digest;
            std::replace(fname.begin(), fname.end(), ':', '-');
            layer.digest = base + "/blobs/" + fname;
        }
    }

    const OllamaLayer * model_layer = nullptr;
    std::vector<const OllamaLayer *> template_layers, params_layers, license_layers;
    for (const auto & layer : manifest.layers) {
        if (layer.media_type == "application/vnd.ollama.image.model") {
            if (model_layer) {
                fprintf(stderr, "\033[31merror: sharded models (multiple weight layers) are not supported yet\033[0m\n");
                return 1;
            }
            model_layer = &layer;
        } else if (layer.media_type == "application/vnd.ollama.image.template") template_layers.push_back(&layer);
        else if (layer.media_type == "application/vnd.ollama.image.params")  params_layers.push_back(&layer);
        else if (layer.media_type == "application/vnd.ollama.image.license") license_layers.push_back(&layer);
    }
    if (!model_layer) {
        fprintf(stderr, "\033[31merror: manifest has no GGUF model layer\033[0m\n");
        return 1;
    }

    std::error_code ec;
    std::filesystem::create_directories(models_dir(), ec);
    if (ec) {
        fprintf(stderr, "\033[31merror: could not create %s\033[0m\n", models_dir().c_str());
        return 1;
    }
    std::string model_path;
    if (source == "ollama") {
        model_path = models_dir() + "/" + friendly + ".gguf";
        if (std::filesystem::exists(model_path)) {
            fprintf(stderr, "\033[31merror: model file already exists at %s (anvil rm %s --yes to replace)\033[0m\n",
                    model_path.c_str(), friendly.c_str());
            return 1;
        }
        printf("Downloading %s (%s)...\n", friendly.c_str(), format_size(model_layer->size).c_str());
        if (pull_blob(ns, name, *model_layer, model_path) != 0) return 1;
    } else {

        const std::string linked = models_dir() + "/" + friendly + ".gguf";
        std::error_code lk;
        std::filesystem::remove(linked, lk);
        lk.clear();
        std::filesystem::create_hard_link(model_layer->digest, linked, lk);
        if (!lk) {
            model_path = linked;
            printf("Linked local blob into anvil store: %s\n", linked.c_str());
        } else {
            model_path = model_layer->digest;
            fprintf(stderr, "\033[33mwarning: could not hardlink blob into anvil store (%s); "
                            "referencing directly — deleting the ollama store will break this model\033[0m\n",
                    lk.message().c_str());
        }
    }

    std::string template_text, license_text, params_json;
    auto read_layer_text = [&](const OllamaLayer & layer) -> std::string {
        if (source == "ollama") {
            const std::string tmp = pull_tmp_path();
            std::error_code e2;
            if (http_download(std::string(OLLAMA_BASE) + "/" + ns + "/" + name + "/blobs/" + layer.digest, tmp) != 0)
                return "";
            std::ifstream f(tmp, std::ios::binary);
            std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::filesystem::remove(tmp, e2);
            return s;
        }
        std::ifstream f(layer.digest, std::ios::binary);
        return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    };
    if (!template_layers.empty()) template_text = read_layer_text(*template_layers[0]);
    if (!params_layers.empty())   params_json  = read_layer_text(*params_layers[0]);
    if (!license_layers.empty())  license_text = read_layer_text(*license_layers[0]);

    return register_pulled(friendly, model_path, source == "ollama" ? "ollama" : "ollama-local",
                           source_id, template_text, license_text, params_json);
}

HWInfo probe_hw();
int derive_ngl(const HWInfo & hw);
bool validate_gguf(const std::string & path);

#ifdef __APPLE__
#endif
#ifdef __linux__
#endif
#ifdef _WIN32
#ifdef _MSC_VER
#pragma comment(lib, "dxgi.lib")
#endif
#endif

#ifdef __APPLE__
static void detect_gpus_macos(HWInfo & hw) {
    CFMutableDictionaryRef matching = IOServiceMatching("IOGPU");
    if (!matching) return;

    mach_port_t mp = MACH_PORT_NULL;
    const kern_return_t main_ret = IOMainPort(MACH_PORT_NULL, &mp);
    if (main_ret != KERN_SUCCESS) { CFRelease(matching); return; }

    io_iterator_t iter = 0;
    const kern_return_t kr = IOServiceGetMatchingServices(mp, matching, &iter);
    if (kr != KERN_SUCCESS) return;

    io_object_t device;
    while ((device = IOIteratorNext(iter)) != 0) {
        GPUInfo gpu;
        gpu.vendor = "Apple";
        gpu.is_discrete = false;
        gpu.vram_mb = 0;
        gpu.name = "Apple GPU (unified " +
                   std::to_string(hw.ram_bytes / (1024ULL * 1024 * 1024)) + " GB)";
        hw.gpus.push_back(gpu);
        IOObjectRelease(device);
    }
    IOObjectRelease(iter);
}
#endif

#ifdef __linux__
static std::string run_cmd(const std::string & cmd) {
    std::string out;
    FILE * pipe = popen(cmd.c_str(), "r");
    if (!pipe) return out;
    char buf[4096];
    size_t n = 0;
    while ((n = fread(buf, 1, sizeof(buf), pipe)) > 0) {
        out.append(buf, n);
    }
    pclose(pipe);
    return out;
}

static void detect_gpus_linux(HWInfo & hw) {

    const std::string out = run_cmd(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null");
    if (!out.empty()) {
        std::istringstream ss(out);
        std::string line;
        while (std::getline(ss, line)) {
            while (!line.empty() && line.back() <= ' ') line.pop_back();
            if (line.empty()) continue;
            GPUInfo gpu;
            gpu.vendor = "NVIDIA";
            gpu.is_discrete = true;

            const auto comma = line.rfind(',');
            if (comma != std::string::npos) {
                std::string vram_str = line.substr(comma + 1);
                while (!vram_str.empty() && vram_str.back() <= ' ') vram_str.pop_back();
                while (!vram_str.empty() && vram_str.front() == ' ') vram_str.erase(vram_str.begin());
                uint64_t vram = 0;
                if (parse_uint64(vram_str, vram)) gpu.vram_mb = vram;
                gpu.name = line.substr(0, comma);
                while (!gpu.name.empty() && gpu.name.back() == ' ') gpu.name.pop_back();
            } else {
                gpu.name = line;
            }
            hw.gpus.push_back(gpu);
        }
    }

    glob_t globbuf;
    if (glob("/sys/class/drm/card*/device/vendor", 0, nullptr, &globbuf) == 0) {
        for (size_t i = 0; i < globbuf.gl_pathc; i++) {
            std::ifstream vf(globbuf.gl_pathv[i]);
            std::string vendor_id;
            std::getline(vf, vendor_id);
            while (!vendor_id.empty() && vendor_id.back() <= ' ') vendor_id.pop_back();

            std::string base(globbuf.gl_pathv[i]);
            const auto pos = base.rfind("/device/vendor");
            if (pos == std::string::npos) continue;
            base = base.substr(0, pos);

            GPUInfo gpu;
            if (vendor_id == "0x1002") {
                gpu.vendor = "AMD";
                gpu.is_discrete = true;
            } else if (vendor_id == "0x8086") {
                gpu.vendor = "Intel";
                gpu.is_discrete = false;
            } else {
                continue;
            }
            std::ifstream df(base + "/device/device");
            std::string did;
            std::getline(df, did);
            gpu.name = gpu.vendor + " GPU (" + did + ")";

            std::ifstream vf2(base + "/device/mem_info_vram_total");
            if (vf2.good()) {
                uint64_t bytes = 0;
                vf2 >> bytes;
                gpu.vram_mb = bytes / (1024 * 1024);
            }
            hw.gpus.push_back(gpu);
        }
        globfree(&globbuf);
    }
}
#endif

#ifdef _WIN32
static void detect_gpus_windows(HWInfo & hw) {
    IDXGIFactory * factory = nullptr;
    if (CreateDXGIFactory(__uuidof(IDXGIFactory), reinterpret_cast<void **>(&factory)) != S_OK) return;
    IDXGIAdapter * adapter = nullptr;
    for (UINT i = 0; factory->EnumAdapters(i, &adapter) != DXGI_ERROR_NOT_FOUND; i++) {
        DXGI_ADAPTER_DESC desc;
        if (adapter->GetDesc(&desc) == S_OK) {
            GPUInfo gpu;
            const int needed = WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, nullptr, 0, nullptr, nullptr);
            if (needed > 1) {
                std::string name_buf(static_cast<size_t>(needed), '\0');
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name_buf.data(), needed, nullptr, nullptr);
                if (!name_buf.empty() && name_buf.back() == '\0') name_buf.pop_back();
                gpu.name = std::move(name_buf);
            } else {
                gpu.name = "Unknown";
            }
            gpu.vram_mb = std::max(desc.DedicatedVideoMemory, desc.SharedSystemMemory) / (1024 * 1024);
            gpu.is_discrete = (desc.VendorId == 0x10DE || desc.VendorId == 0x1002);
            if (desc.VendorId == 0x10DE)      gpu.vendor = "NVIDIA";
            else if (desc.VendorId == 0x1002) gpu.vendor = "AMD";
            else if (desc.VendorId == 0x8086) gpu.vendor = "Intel";
            else                              gpu.vendor = "Other";
            hw.gpus.push_back(gpu);
        }
        adapter->Release();
    }
    factory->Release();
}
#endif

HWInfo probe_hw() {
    HWInfo hw;

#if defined(__APPLE__)
    hw.os = "macos";
#elif defined(__linux__)
    hw.os = "linux";
#elif defined(_WIN32)
    hw.os = "windows";
#else
    hw.os = "unknown";
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    hw.arch = "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    hw.arch = "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    hw.arch = "i386";
#else
    hw.arch = "unknown";
#endif

    hw.cpu_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (hw.cpu_threads <= 0) hw.cpu_threads = 0;

#ifdef __APPLE__
    {
        std::string buf;
        size_t len = 0;
        if (sysctlbyname("machdep.cpu.brand_string", nullptr, &len, nullptr, 0) == 0 && len > 0) {
            buf.resize(len);
            if (sysctlbyname("machdep.cpu.brand_string", &buf[0], &len, nullptr, 0) == 0) {
                while (!buf.empty() && buf.back() == '\0') buf.pop_back();
                hw.cpu = buf;
                hw.apple_silicon = (hw.cpu.find("Apple") != std::string::npos);
            }
        }
    }
    {
        uint64_t ram = 0;
        size_t len = sizeof(ram);
        if (sysctlbyname("hw.memsize", &ram, &len, nullptr, 0) == 0) {
            hw.ram_bytes = ram;
        }
    }
    detect_gpus_macos(hw);
#elif defined(__linux__)
    {
        std::ifstream f("/proc/cpuinfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("model name") != std::string::npos) {
                const auto pos = line.find(':');
                if (pos != std::string::npos) {
                    hw.cpu = line.substr(pos + 2);
                    while (!hw.cpu.empty() && hw.cpu.back() <= ' ') hw.cpu.pop_back();
                }
                break;
            }
        }
    }
    {
        std::ifstream f("/proc/meminfo");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("MemTotal") != std::string::npos) {
                std::istringstream iss(line);
                std::string key;
                uint64_t kb = 0;
                std::string unit;
                if ((iss >> key >> kb >> unit) && key == "MemTotal" && unit == "kB") {
                    hw.ram_bytes = kb * 1024;
                }
                break;
            }
        }
    }
    detect_gpus_linux(hw);
#elif defined(_WIN32)
    {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        hw.arch = (si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64) ? "aarch64" : "x86_64";
        hw.cpu_threads = static_cast<int>(si.dwNumberOfProcessors);
        hw.cpu = "Windows CPU";
    }
    {
        MEMORYSTATUSEX ms;
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms)) {
            hw.ram_bytes = ms.ullTotalPhys;
        }
    }
    detect_gpus_windows(hw);
#endif
    return hw;
}

int derive_ngl(const HWInfo & hw) {

    if (hw.apple_silicon) return -1;
    if (!hw.gpus.empty()) return -1;
    return 0;
}

bool validate_gguf(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[4] = {0, 0, 0, 0};
    f.read(magic, 4);
    return f.gcount() == 4 && memcmp(magic, "GGUF", 4) == 0;
}

std::string gguf_check_error(const std::string & path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return "'" + path + "': no such file (was it moved or deleted?)";
    }
    return "'" + path + "' is not a valid GGUF file (missing GGUF magic; is the download complete?)";
}

AnvilConfig run_setup_tui(const HWInfo & hw, int max_ctx);

AnvilConfig run_setup_tui(const HWInfo & hw, int max_ctx) {

    const bool known_ctx = max_ctx > 0;
    if (max_ctx <= 0) max_ctx = 131072;
    AnvilConfig cfg;
    cfg.ngl = derive_ngl(hw);
    cfg.n_ctx = known_ctx ? max_ctx : 4096;

    std::vector<int> ctx_options;
    for (int c = 1; c > 0 && c <= max_ctx; c *= 2) {
        if (c > INT32_MAX / 2) break;
        ctx_options.push_back(c);
    }
    ctx_options.push_back(0);
    const int custom_idx = static_cast<int>(ctx_options.size()) - 1;
    int ctx_sel = 0;
    for (size_t i = 0; i < ctx_options.size(); i++) {
        if (ctx_options[i] == cfg.n_ctx) { ctx_sel = static_cast<int>(i); break; }
    }

    std::vector<std::string> ctx_labels;
    for (const int c : ctx_options) {
        if (c == 0) ctx_labels.push_back("Custom...");
        else if (c >= 1024) ctx_labels.push_back(std::to_string(c / 1024) + "K tokens");
        else ctx_labels.push_back(std::to_string(c) + " tokens");
    }

    std::vector<std::string> kv_preset_labels;
    for (int i = 0; i < KV_PRESETS_COUNT; i++) kv_preset_labels.push_back(KV_PRESETS[i].label);

    std::vector<std::string> kv_type_labels;
    for (int i = 0; i < KV_OPTIONS_COUNT; i++) kv_type_labels.push_back(KV_OPTIONS[i].label);

    const std::vector<std::string> fa_labels   = {"on (recommended)", "off"};
    const std::vector<std::string> temp_labels = {"0.7 (focused)", "0.8 (balanced)", "0.9 (creative)", "1.0 (wild)"};

    int kv_preset_sel = 0;
    int kv_k_sel      = 1;
    int kv_v_sel      = 3;
    int fa_sel        = 0;
    int temp_sel      = 1;
    bool custom_kv    = false;
    int menu_idx      = 0;

    std::vector<std::string> hw_lines;
    hw_lines.push_back("CPU  : " + hw.cpu);
    hw_lines.push_back("RAM  : " + std::to_string(hw.ram_bytes / (1024ULL * 1024 * 1024)) + " GB");
    hw_lines.push_back("Threads: " + std::to_string(hw.cpu_threads));
    for (const auto & gpu : hw.gpus) {
        hw_lines.push_back("GPU  : " + gpu.name + " (" + std::to_string(gpu.vram_mb) + " MB)");
    }

    auto screen = ftxui::ScreenInteractive::FitComponent();
    auto component = ftxui::Renderer([&]() {
        ftxui::Elements rows;
        rows.push_back(ftxui::text(ANVIL_LOGO) | ftxui::bold | ftxui::color(ftxui::Color::Yellow));
        rows.push_back(ftxui::text("  First-time setup  v" + std::string(ANVIL_VERSION)) | ftxui::bold | ftxui::color(ftxui::Color::Cyan));
        rows.push_back(ftxui::text(" "));
        rows.push_back(ftxui::text("  Detected hardware:") | ftxui::bold);
        for (const auto & line : hw_lines) rows.push_back(ftxui::text("    " + line));
        rows.push_back(ftxui::text(" "));

        auto render_setting = [&](const std::string & label, const std::vector<std::string> & opts,
                                  int sel, int idx, int indent = 2) {
            const std::string pad(static_cast<size_t>(indent), ' ');
            const bool active = (idx == menu_idx);
            auto lbl = ftxui::text(pad + label);
            if (active) lbl = lbl | ftxui::bold | ftxui::color(ftxui::Color::Cyan);
            else        lbl = lbl | ftxui::bold;
            rows.push_back(lbl);
            for (int i = 0; i < static_cast<int>(opts.size()); i++) {
                const bool selected = (i == sel);
                auto prefix = selected
                    ? ftxui::text(pad + "  ▸ ") | ftxui::bold | ftxui::color(ftxui::Color::Green)
                    : ftxui::text(pad + "  · ");
                auto label_el = ftxui::text(opts[static_cast<size_t>(i)]);
                if (selected && active) label_el = label_el | ftxui::bold | ftxui::color(ftxui::Color::Green);
                else if (selected)      label_el = label_el | ftxui::color(ftxui::Color::Green);
                rows.push_back(ftxui::hbox({prefix, label_el}));
            }
            rows.push_back(ftxui::text(" "));
        };

        render_setting("Context size", ctx_labels, ctx_sel, 0);
        render_setting("KV cache compression", kv_preset_labels, kv_preset_sel, 1);
        if (custom_kv) {
            render_setting("  K cache type", kv_type_labels, kv_k_sel, 4, 4);
            render_setting("  V cache type", kv_type_labels, kv_v_sel, 5, 4);
        }
        render_setting("Flash attention", fa_labels, fa_sel, 2);
        render_setting("Temperature", temp_labels, temp_sel, 3);

        std::string kv_summary;
        if (!custom_kv) {
            const int ki = KV_PRESETS[kv_preset_sel].k_idx;
            const int vi = KV_PRESETS[kv_preset_sel].v_idx;
            if (ki >= 0 && vi >= 0)
                kv_summary = "  Active: K=" + std::string(KV_OPTIONS[ki].short_name) +
                             " V=" + std::string(KV_OPTIONS[vi].short_name);
        } else {
            kv_summary = "  Active: K=" + std::string(KV_OPTIONS[kv_k_sel].short_name) +
                         " V=" + std::string(KV_OPTIONS[kv_v_sel].short_name);
        }
        rows.push_back(ftxui::text(kv_summary) | ftxui::dim);
        rows.push_back(ftxui::text(" "));
        rows.push_back(ftxui::text("  Tab switch  ↑/↓ change  Enter confirm  q cancel") | ftxui::dim);

        return ftxui::vbox(std::move(rows));
    });

    auto wrapped = component | ftxui::CatchEvent([&](ftxui::Event e) {
        if (e == ftxui::Event::Character('q')) { screen.Exit(); return true; }
        if (e == ftxui::Event::Return) { screen.Exit(); return true; }

        const int menu_count = custom_kv ? 6 : 4;
        if (e == ftxui::Event::Tab) {
            menu_idx = (menu_idx + 1) % menu_count;
            if (!custom_kv && menu_idx >= 4) menu_idx = 0;
            return true;
        }
        if (e == ftxui::Event::TabReverse) {
            menu_idx = (menu_idx - 1 + menu_count) % menu_count;
            if (!custom_kv && menu_idx >= 4) menu_idx = 3;
            return true;
        }
        auto cycle = [](int & sel, int count, int dir) {
            sel = (sel + dir + count) % count;
        };
        if (e == ftxui::Event::ArrowUp) {
            switch (menu_idx) {
                case 0: cycle(ctx_sel, static_cast<int>(ctx_labels.size()), -1); break;
                case 1:
                    cycle(kv_preset_sel, KV_PRESETS_COUNT, -1);
                    custom_kv = (kv_preset_sel == KV_PRESETS_COUNT - 1);
                    break;
                case 2: cycle(fa_sel, static_cast<int>(fa_labels.size()), -1); break;
                case 3: cycle(temp_sel, static_cast<int>(temp_labels.size()), -1); break;
                case 4: cycle(kv_k_sel, KV_OPTIONS_COUNT, -1); break;
                case 5: cycle(kv_v_sel, KV_OPTIONS_COUNT, -1); break;
                default: break;
            }
            return true;
        }
        if (e == ftxui::Event::ArrowDown) {
            switch (menu_idx) {
                case 0: cycle(ctx_sel, static_cast<int>(ctx_labels.size()), 1); break;
                case 1:
                    cycle(kv_preset_sel, KV_PRESETS_COUNT, 1);
                    custom_kv = (kv_preset_sel == KV_PRESETS_COUNT - 1);
                    break;
                case 2: cycle(fa_sel, static_cast<int>(fa_labels.size()), 1); break;
                case 3: cycle(temp_sel, static_cast<int>(temp_labels.size()), 1); break;
                case 4: cycle(kv_k_sel, KV_OPTIONS_COUNT, 1); break;
                case 5: cycle(kv_v_sel, KV_OPTIONS_COUNT, 1); break;
                default: break;
            }
            return true;
        }
        return false;
    });

    screen.Loop(wrapped);

    if (ctx_sel == custom_idx) {
        printf("\033[?25h");
        printf("Enter context size (tokens): ");
        fflush(stdout);
        std::string input;
        std::getline(std::cin, input);
        if (!parse_int(input, cfg.n_ctx) || cfg.n_ctx <= 0 || cfg.n_ctx > MAX_CTX) {
            fprintf(stderr, "\033[33mwarning: invalid context size '%s', using auto\033[0m\n", input.c_str());
            cfg.n_ctx = max_ctx > 0 ? max_ctx : 0;
        }
    } else {
        cfg.n_ctx = ctx_options[static_cast<size_t>(ctx_sel)];
    }

    if (custom_kv) {
        cfg.type_k = KV_OPTIONS[kv_k_sel].type;
        cfg.type_v = KV_OPTIONS[kv_v_sel].type;
    } else {
        const int ki = KV_PRESETS[kv_preset_sel].k_idx;
        const int vi = KV_PRESETS[kv_preset_sel].v_idx;
        cfg.type_k = KV_OPTIONS[ki].type;
        cfg.type_v = KV_OPTIONS[vi].type;
    }
    cfg.flash_attn = (fa_sel == 0);
    const float temps[] = {0.7f, 0.8f, 0.9f, 1.0f};
    cfg.temp = temps[temp_sel];

    return cfg;
}

int run_chat(const CliArgs & cli, AnvilConfig cfg, const HWInfo & hw,
             std::vector<ModelEntry> & models);

#ifdef _WIN32
typedef SOCKET anvil_sock_t;
#define ANVIL_SOCK_INVALID INVALID_SOCKET
#define ANVIL_SOCK_CLOSE closesocket
#define ANVIL_SOCK_ERR SOCKET_ERROR
#else
typedef int anvil_sock_t;
#define ANVIL_SOCK_INVALID (-1)
#define ANVIL_SOCK_CLOSE ::close
#define ANVIL_SOCK_ERR (-1)
#endif

static std::string g_serve_api_key;
static std::mutex g_serve_infer_lock;

static std::string token_to_str(const llama_vocab * vocab, llama_token token);
static llama_sampler * build_sampler_chain(const llama_vocab * vocab, const AnvilConfig & cfg,
        bool grammar_active, const std::string & grammar_src, int seed = -1);
static std::vector<llama_token> tokenize_render(const llama_vocab * vocab, const std::string & text);

static int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string serve_id(const char * prefix) {
    static std::atomic<uint64_t> counter{0};
    char buf[96];
    std::snprintf(buf, sizeof(buf), "%s%llx%llx",
                  prefix,
                  static_cast<unsigned long long>(now_unix()),
                  static_cast<unsigned long long>(counter.fetch_add(1)));
    return std::string(buf);
}

static std::vector<llama_token> serve_tokenize(const llama_vocab * vocab, const std::string & text, bool parse_special) {
    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               nullptr, 0, false, parse_special);
    if (n == INT32_MIN) return {};
    if (n < 0) n = -n;
    if (n <= 0) return {};
    std::vector<llama_token> toks(static_cast<size_t>(n));
    const int32_t m = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                     toks.data(), n, false, parse_special);
    if (m < 0) return {};
    toks.resize(static_cast<size_t>(m));
    return toks;
}

static bool serve_decode_slot(llama_context * ctx, llama_seq_id seq, llama_pos start_pos,
                              const std::vector<llama_token> & toks) {
    const int32_t chunk = static_cast<int32_t>(llama_n_batch(ctx));
    for (size_t i = 0; i < toks.size(); i += static_cast<size_t>(chunk)) {
        const size_t nn = std::min<size_t>(static_cast<size_t>(chunk), toks.size() - i);
        llama_batch batch = llama_batch_init(static_cast<int32_t>(nn), 0, 1);
        batch.n_tokens = static_cast<int32_t>(nn);
        for (size_t j = 0; j < nn; j++) {
            batch.token[j] = toks[i + j];
            batch.pos[j] = start_pos + static_cast<llama_pos>(i + j);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = seq;
            batch.logits[j] = 1;
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) return false;
    }
    return true;
}

static bool serve_decode_one(llama_context * ctx, llama_seq_id seq, llama_pos pos, llama_token id) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.token[0] = id;
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = seq;
    batch.logits[0] = 1;
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

struct ServeSlot {
    llama_seq_id id = 0;
    std::vector<llama_token> prompt_tokens;
    llama_pos n_past = 0;
    bool busy = false;
    int64_t t_last_used = 0;
};

struct ServeSlots {
    llama_context * ctx = nullptr;
    llama_memory_t mem = nullptr;
    common_context_seq_rm_type rm_type = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    std::vector<ServeSlot> slots;
    size_t min_prefix = 8;

    llama_seq_id emb_seq() const { return static_cast<llama_seq_id>(slots.size()); }

    void init(llama_context * c, llama_memory_t m, int n) {
        ctx = c;
        mem = m;
        rm_type = common_context_can_seq_rm(ctx);
        slots.clear();
        for (int i = 0; i < n; i++) {
            ServeSlot s;
            s.id = static_cast<llama_seq_id>(i);
            s.t_last_used = now_unix();
            slots.push_back(s);
        }
    }

    void reset_all() {
        for (auto & s : slots) {
            llama_memory_seq_rm(mem, s.id, -1, -1);
            s.prompt_tokens.clear();
            s.n_past = 0;
            s.busy = false;
            s.t_last_used = now_unix();
        }
        if (rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            llama_memory_clear(mem, true);
        }
    }

    static size_t common_prefix(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) n++;
        return n;
    }

    ServeSlot * acquire(const std::vector<llama_token> & toks) {
        ServeSlot * best = nullptr;
        size_t best_lcp = 0;
        for (auto & s : slots) {
            if (s.busy || s.prompt_tokens.empty()) continue;
            const size_t lcp = common_prefix(s.prompt_tokens, toks);
            if (lcp >= min_prefix && lcp > best_lcp) {
                best_lcp = lcp;
                best = &s;
            }
        }
        if (best) {
            best->busy = true;
            return best;
        }
        ServeSlot * lru = nullptr;
        for (auto & s : slots) {
            if (s.busy) continue;
            if (!lru || s.t_last_used < lru->t_last_used) lru = &s;
        }
        if (lru) lru->busy = true;
        return lru;
    }

    void release(ServeSlot & s) {
        s.busy = false;
        s.t_last_used = now_unix();
    }

    bool prepare(ServeSlot & s, const std::vector<llama_token> & toks) {
        size_t n_common = common_prefix(s.prompt_tokens, toks);
        if (rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            for (const auto & o : slots) {
                if (o.id == s.id || o.prompt_tokens.empty()) continue;
                const size_t lcp = common_prefix(o.prompt_tokens, toks);
                if (lcp > n_common && lcp >= min_prefix) {
                    llama_memory_seq_cp(mem, o.id, s.id, 0, static_cast<llama_pos>(lcp));
                    n_common = lcp;
                }
            }
        }
        if (rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
            llama_memory_clear(mem, true);
            for (auto & o : slots) {
                o.prompt_tokens.clear();
                o.n_past = 0;
            }
            n_common = 0;
        } else if (n_common < static_cast<size_t>(s.n_past)) {
            const bool can_partial = rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
                                     rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;
            if (can_partial) {
                if (!llama_memory_seq_rm(mem, s.id, static_cast<llama_pos>(n_common), -1)) {
                    llama_memory_seq_rm(mem, s.id, -1, -1);
                    n_common = 0;
                }
            } else {
                llama_memory_seq_rm(mem, s.id, -1, -1);
                n_common = 0;
            }
        }
        if (n_common < toks.size()) {
            const std::vector<llama_token> suffix(toks.begin() + static_cast<long>(n_common), toks.end());
            if (!serve_decode_slot(ctx, s.id, static_cast<llama_pos>(n_common), suffix)) return false;
        } else {
            std::vector<llama_token> last{toks.back()};
            if (!serve_decode_slot(ctx, s.id, static_cast<llama_pos>(n_common) - 1, last)) return false;
        }
        s.prompt_tokens = toks;
        s.n_past = static_cast<llama_pos>(toks.size());
        return true;
    }

    bool decode_token(ServeSlot & s, llama_token id) {
        if (!serve_decode_one(ctx, s.id, s.n_past, id)) return false;
        s.n_past++;
        return true;
    }

    void extend(ServeSlot & s, const std::vector<llama_token> & gen) {
        if (gen.empty()) return;
        s.prompt_tokens.insert(s.prompt_tokens.end(), gen.begin(), gen.end());
    }
};

struct SlotHolder {
    ServeSlots & pool;
    ServeSlot * slot;
    ~SlotHolder() { if (slot) pool.release(*slot); }
};

struct ServeGenCtx {
    llama_context * ctx = nullptr;
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
    llama_memory_t mem = nullptr;
    common_chat_templates * tmpls = nullptr;
    AnvilConfig cfg;
    std::string model_id;
    ServeSlots slots;
};

struct ServeResult {
    common_chat_msg msg;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    std::string finish_reason = "stop";
};

static std::string serve_tool_choice(const nlohmann::ordered_json & body) {
    if (!body.contains("tool_choice") || body["tool_choice"].is_null()) return "auto";
    const auto & tc = body["tool_choice"];
    if (tc.is_string()) return tc.get<std::string>();
    if (tc.is_object()) {
        const std::string type = tc.value("type", std::string("auto"));
        if (type == "any" || type == "tool") return "required";
        return type;
    }
    return "auto";
}

static bool serve_chat(ServeGenCtx & g, const nlohmann::ordered_json & body,
                       const std::function<bool(const common_chat_msg_diff &)> & emit,
                       ServeResult & out, std::string & err,
                       const volatile sig_atomic_t & stop,
                       const std::function<void(int)> & on_prompt = {}) {
    common_chat_templates_inputs inputs;
    std::string generated;
    try {
        if (!body.contains("messages") || !body["messages"].is_array()) {
            err = "messages is required";
            return false;
        }
        inputs.messages = common_chat_msgs_parse_oaicompat(body["messages"]);
        if (inputs.messages.empty()) {
            err = "messages is empty";
            return false;
        }
        for (const auto & m : inputs.messages) {
            if (m.contains_media()) {
                err = "image content is not supported in serve yet";
                return false;
            }
        }
        if (body.contains("tools") && body["tools"].is_array()) {
            inputs.tools = common_chat_tools_parse_oaicompat(body["tools"]);
        }
        inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(serve_tool_choice(body));
        if (body.contains("response_format") && body["response_format"].is_object()) {
            const auto & rf = body["response_format"];
            const std::string type = rf.value("type", std::string());
            if (type == "json_object") {
                inputs.json_schema = "{}";
            } else if (type == "json_schema" && rf.contains("json_schema")) {
                const auto & js = rf["json_schema"];
                inputs.json_schema = js.contains("schema") ? js["schema"].dump() : js.dump();
            }
        }
        if (body.contains("grammar")) inputs.grammar = body["grammar"].get<std::string>();
        inputs.parallel_tool_calls = body.value("parallel_tool_calls", true);
        inputs.enable_thinking = body.value("enable_thinking", true);
        const std::string rf_name = body.value("reasoning_format", std::string("auto"));
        inputs.reasoning_format = common_reasoning_format_from_name(rf_name);
        const auto chat_params = common_chat_templates_apply(g.tmpls, inputs);
        if (chat_params.prompt.empty()) {
            err = "chat template produced an empty prompt";
            return false;
        }
        std::vector<llama_token> toks = serve_tokenize(g.vocab, chat_params.prompt, true);
        if (toks.empty()) {
            err = "prompt tokenization failed";
            return false;
        }
        ServeSlot * slot = g.slots.acquire(toks);
        if (!slot) {
            err = "all slots busy";
            return false;
        }
        SlotHolder holder{g.slots, slot};
        if (!g.slots.prepare(*slot, toks)) {
            err = "prompt decode failed";
            return false;
        }
        out.prompt_tokens = static_cast<int>(toks.size());
        if (on_prompt) on_prompt(out.prompt_tokens);
        std::vector<llama_token> gen_toks;

        AnvilConfig cfg = g.cfg;
        if (body.contains("temperature") && body["temperature"].is_number()) cfg.temp = body["temperature"].get<float>();
        if (body.contains("top_p") && body["top_p"].is_number()) cfg.top_p = body["top_p"].get<float>();
        if (body.contains("top_k") && body["top_k"].is_number_integer()) cfg.top_k = body["top_k"].get<int>();
        if (body.contains("seed") && body["seed"].is_number_integer()) cfg.seed = body["seed"].get<int>();
        if (body.contains("repeat_penalty") && body["repeat_penalty"].is_number()) cfg.repeat_penalty = body["repeat_penalty"].get<float>();

        LlamaSampler smpl(build_sampler_chain(g.vocab, cfg, false, "", cfg.seed));
        if (!smpl) {
            err = "sampler init failed";
            return false;
        }
        llama_sampler * gs = nullptr;
        if (!chat_params.grammar.empty()) {
            if (chat_params.grammar_lazy) {
                std::vector<std::string> trigger_patterns;
                std::vector<llama_token> trigger_tokens;
                for (const auto & trigger : chat_params.grammar_triggers) {
                    switch (trigger.type) {
                        case COMMON_GRAMMAR_TRIGGER_TYPE_WORD: {
                            trigger_patterns.push_back(regex_escape(trigger.value));
                            break;
                        }
                        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN: {
                            trigger_patterns.push_back(trigger.value);
                            break;
                        }
                        case COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL: {
                            const auto & pattern = trigger.value;
                            std::string anchored = "^$";
                            if (!pattern.empty()) {
                                anchored = (pattern.front() != '^' ? "^" : "")
                                    + pattern
                                    + (pattern.back() != '$' ? "$" : "");
                            }
                            trigger_patterns.push_back(anchored);
                            break;
                        }
                        case COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN: {
                            trigger_tokens.push_back(trigger.token);
                            break;
                        }
                    }
                }
                std::vector<const char *> trigger_patterns_c;
                trigger_patterns_c.reserve(trigger_patterns.size());
                for (const auto & regex : trigger_patterns) trigger_patterns_c.push_back(regex.c_str());
                gs = llama_sampler_init_grammar_lazy_patterns(g.vocab, chat_params.grammar.c_str(), "root",
                        trigger_patterns_c.data(), trigger_patterns_c.size(),
                        trigger_tokens.data(), trigger_tokens.size());
            } else {
                gs = llama_sampler_init_grammar(g.vocab, chat_params.grammar.c_str(), "root");
            }
        }
        LlamaSampler gs_guard(gs);
        llama_sampler_reset(smpl.get());
        bool grammar_alive = gs != nullptr;
        if (gs && !chat_params.grammar_lazy) {
            const auto prefill = serve_tokenize(g.vocab, chat_params.generation_prompt, true);
            for (const auto & t : prefill) {
                try { llama_sampler_accept(gs, t); }
                catch (...) { break; }
            }
        }
        std::vector<llama_token_data> cand;
        cand.reserve(static_cast<size_t>(llama_vocab_n_tokens(g.vocab)));

        std::vector<std::string> stops;
        if (body.contains("stop")) {
            const auto & s = body["stop"];
            if (s.is_string()) stops.push_back(s.get<std::string>());
            else if (s.is_array()) {
                for (const auto & x : s) if (x.is_string()) stops.push_back(x.get<std::string>());
            }
        }
        for (const auto & s : chat_params.additional_stops) stops.push_back(s);

        const int max_tokens = body.value("max_tokens", body.value("max_completion_tokens", -1));
        std::set<llama_token> preserved;
        for (const auto & s : chat_params.preserved_tokens) {
            const auto ptoks = serve_tokenize(g.vocab, s, true);
            if (!ptoks.empty()) preserved.insert(ptoks[0]);
        }

        common_chat_parser_params pparams(chat_params);
        pparams.parse_tool_calls = !inputs.tools.empty();
        if (!chat_params.parser.empty()) pparams.parser.load(chat_params.parser);

        common_chat_msg msg_prv;
        std::vector<std::string> ids_cache;
        auto gen_id = []() { return serve_id("call_"); };

        std::set<size_t> sent_tool_names;
        auto apply_diffs = [&](bool is_partial) {
            const auto new_msg = common_chat_parse(generated, is_partial, pparams);
            if (new_msg.empty() || new_msg == msg_prv) return;
            common_chat_msg nxt = new_msg;
            nxt.set_tool_call_ids(ids_cache, gen_id);
            if (!emit) {
                msg_prv = nxt;
                return;
            }
            const auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv, nxt);
            msg_prv = nxt;
            for (auto d : all_diffs) {
                if (d.tool_call_index != std::string::npos) {
                    const size_t i = d.tool_call_index;
                    if (i < msg_prv.tool_calls.size() && !msg_prv.tool_calls[i].name.empty() &&
                        !sent_tool_names.count(i) && (d.tool_call_index != i || !d.tool_call_delta.arguments.empty())) {
                        common_chat_msg_diff header;
                        header.tool_call_index = static_cast<size_t>(i);
                        header.tool_call_delta.id = msg_prv.tool_calls[i].id;
                        header.tool_call_delta.name = msg_prv.tool_calls[i].name;
                        if (!emit(header)) return;
                        sent_tool_names.insert(i);
                    }
                    if (sent_tool_names.count(i)) d.tool_call_delta.name = "";
                }
                if (!emit(d)) return;
            }
        };

        bool length_hit = false;
        bool stop_hit = false;
        while (true) {
            if (stop) break;
            if (max_tokens > 0 && out.completion_tokens >= max_tokens) {
                length_hit = true;
                break;
            }
            const float * logits = llama_get_logits_ith(g.ctx, -1);
            if (!logits) { err = "logits unavailable"; return false; }
            const int32_t n_vocab = llama_vocab_n_tokens(g.vocab);
            cand.resize(static_cast<size_t>(n_vocab));
            for (llama_token i = 0; i < n_vocab; i++) {
                cand[static_cast<size_t>(i)] = { i, logits[i], 0.0f };
            }
            llama_token_data_array cur_p = { cand.data(), cand.size(), -1, false };
            llama_sampler_apply(smpl.get(), &cur_p);
            if (cur_p.selected < 0) { err = "sampling produced no token"; return false; }
            llama_token id = cur_p.data[cur_p.selected].id;
            if (gs && grammar_alive) {
                llama_token_data single = { id, 1.0f, 0.0f };
                llama_token_data_array single_arr = { &single, 1, -1, false };
                llama_sampler_apply(gs, &single_arr);
                if (single_arr.data[0].logit == -INFINITY) {
                    cand.resize(static_cast<size_t>(n_vocab));
                    for (llama_token i = 0; i < n_vocab; i++) {
                        cand[static_cast<size_t>(i)] = { i, logits[i], 0.0f };
                    }
                    cur_p = { cand.data(), cand.size(), -1, false };
                    llama_sampler_apply(gs, &cur_p);
                    llama_sampler_apply(smpl.get(), &cur_p);
                    if (cur_p.selected < 0) { err = "sampling produced no token"; return false; }
                    id = cur_p.data[cur_p.selected].id;
                }
            }
            llama_sampler_accept(smpl.get(), id);
            if (gs) {
                try { llama_sampler_accept(gs, id); }
                catch (...) { grammar_alive = false; }
            }
            if (llama_vocab_is_eog(g.vocab, id)) break;
            if (llama_vocab_is_control(g.vocab, id) && !preserved.count(id)) break;
            const std::string piece = token_to_str(g.vocab, id);
            generated += piece;
            out.completion_tokens++;
            bool hit = false;
            for (const auto & s : stops) {
                if (s.empty() || generated.size() < s.size()) continue;
                if (generated.compare(generated.size() - s.size(), s.size(), s) == 0) {
                    generated.erase(generated.size() - s.size());
                    hit = true;
                    stop_hit = true;
                    break;
                }
            }
            if (hit) break;
            apply_diffs(true);
            gen_toks.push_back(id);
            if (!g.slots.decode_token(*slot, id)) {
                err = "decode error";
                return false;
            }
        }
        g.slots.extend(*slot, gen_toks);

        apply_diffs(false);
        out.msg = common_chat_parse(generated, false, pparams);
        out.msg.set_tool_call_ids(ids_cache, gen_id);

        if (length_hit) out.finish_reason = "length";
        else if (stop_hit) out.finish_reason = "stop";
        else if (!out.msg.tool_calls.empty()) out.finish_reason = "tool_calls";
        else out.finish_reason = "stop";
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

static bool serve_completions(ServeGenCtx & g, const nlohmann::ordered_json & body,
                              ServeResult & out, std::string & err,
                              const volatile sig_atomic_t & stop) {
    try {
        const std::string prompt = body.value("prompt", std::string());
        if (prompt.empty()) {
            err = "prompt is required";
            return false;
        }
        std::vector<llama_token> toks = serve_tokenize(g.vocab, prompt, false);
        if (toks.empty()) {
            err = "prompt tokenization failed";
            return false;
        }
        ServeSlot * slot = g.slots.acquire(toks);
        if (!slot) {
            err = "all slots busy";
            return false;
        }
        SlotHolder holder{g.slots, slot};
        if (!g.slots.prepare(*slot, toks)) {
            err = "prompt decode failed";
            return false;
        }
        out.prompt_tokens = static_cast<int>(toks.size());
        std::vector<llama_token> gen_toks;

        AnvilConfig cfg = g.cfg;
        if (body.contains("temperature") && body["temperature"].is_number()) cfg.temp = body["temperature"].get<float>();
        if (body.contains("top_p") && body["top_p"].is_number()) cfg.top_p = body["top_p"].get<float>();
        if (body.contains("seed") && body["seed"].is_number_integer()) cfg.seed = body["seed"].get<int>();
        LlamaSampler smpl(build_sampler_chain(g.vocab, cfg, false, "", cfg.seed));
        if (!smpl) {
            err = "sampler init failed";
            return false;
        }
        llama_sampler_reset(smpl.get());

        std::vector<std::string> stops;
        if (body.contains("stop")) {
            const auto & s = body["stop"];
            if (s.is_string()) stops.push_back(s.get<std::string>());
            else if (s.is_array()) {
                for (const auto & x : s) if (x.is_string()) stops.push_back(x.get<std::string>());
            }
        }
        const int max_tokens = body.value("max_tokens", 16);
        std::string generated;
        bool length_hit = false;
        while (true) {
            if (stop) break;
            if (out.completion_tokens >= max_tokens) {
                length_hit = true;
                break;
            }
            const llama_token id = llama_sampler_sample(smpl.get(), g.ctx, -1);
            if (llama_vocab_is_eog(g.vocab, id)) break;
            const std::string piece = token_to_str(g.vocab, id);
            generated += piece;
            out.completion_tokens++;
            bool hit = false;
            for (const auto & s : stops) {
                if (s.empty() || generated.size() < s.size()) continue;
                if (generated.compare(generated.size() - s.size(), s.size(), s) == 0) {
                    generated.erase(generated.size() - s.size());
                    hit = true;
                    break;
                }
            }
            if (hit) break;
            gen_toks.push_back(id);
            if (!g.slots.decode_token(*slot, id)) {
                err = "decode error";
                return false;
            }
        }
        g.slots.extend(*slot, gen_toks);
        out.msg.role = "assistant";
        out.msg.content = generated;
        out.finish_reason = length_hit ? "length" : "stop";
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

static bool serve_embeddings(ServeGenCtx & g, const nlohmann::ordered_json & body,
                             nlohmann::ordered_json & out, std::string & err,
                             const volatile sig_atomic_t & stop) {
    try {
        std::vector<std::string> inputs;
        if (body.contains("input")) {
            const auto & in = body["input"];
            if (in.is_string()) inputs.push_back(in.get<std::string>());
            else if (in.is_array()) {
                for (const auto & x : in) if (x.is_string()) inputs.push_back(x.get<std::string>());
            }
        }
        if (inputs.empty()) {
            err = "input is required";
            return false;
        }
        const bool is_encoder = llama_model_has_encoder(g.model);
        const int32_t n_embd = llama_model_n_embd(g.model);
        nlohmann::ordered_json data = nlohmann::ordered_json::array();
        int total = 0;
        const llama_seq_id eseq = g.slots.emb_seq();
        for (size_t i = 0; i < inputs.size(); i++) {
            if (stop) break;
            if (!llama_memory_seq_rm(g.mem, eseq, -1, -1)) {
                err = "embedding cache clear failed";
                return false;
            }
            std::vector<llama_token> toks = serve_tokenize(g.vocab, inputs[i], false);
            if (toks.empty()) {
                err = "input tokenization failed";
                return false;
            }
            llama_batch batch = llama_batch_init(static_cast<int32_t>(toks.size()), 0, 1);
            batch.n_tokens = static_cast<int32_t>(toks.size());
            for (size_t j = 0; j < toks.size(); j++) {
                batch.token[j] = toks[j];
                batch.pos[j] = static_cast<llama_pos>(j);
                batch.n_seq_id[j] = 1;
                batch.seq_id[j][0] = eseq;
                batch.logits[j] = 1;
            }
            llama_set_embeddings(g.ctx, true);
            const int rc = is_encoder ? llama_encode(g.ctx, batch) : llama_decode(g.ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) {
                err = "embedding decode failed";
                return false;
            }
            const float * embd = llama_get_embeddings_seq(g.ctx, eseq);
            if (!embd) embd = llama_get_embeddings(g.ctx);
            if (!embd) {
                err = "embeddings unavailable for this model";
                return false;
            }
            nlohmann::ordered_json vec = nlohmann::ordered_json::array();
            for (int32_t j = 0; j < n_embd; j++) vec.push_back(embd[j]);
            nlohmann::ordered_json item;
            item["object"] = "embedding";
            item["index"] = i;
            item["embedding"] = std::move(vec);
            data.push_back(std::move(item));
            total += static_cast<int>(toks.size());
        }
        out["object"] = "list";
        out["data"] = std::move(data);
        out["model"] = g.model_id;
        out["usage"] = {{"prompt_tokens", total}, {"total_tokens", total}};
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

static nlohmann::ordered_json anthropic_to_oai(const nlohmann::ordered_json & body) {
    nlohmann::ordered_json oai;
    nlohmann::ordered_json messages = nlohmann::ordered_json::array();
    if (body.contains("system")) {
        const auto & sys = body["system"];
        std::string text;
        if (sys.is_string()) text = sys.get<std::string>();
        else if (sys.is_array()) {
            for (const auto & b : sys) {
                if (b.value("type", std::string()) == "text") text += b.value("text", std::string());
            }
        }
        if (!text.empty()) messages.push_back({{"role", "system"}, {"content", text}});
    }
    if (body.contains("messages") && body["messages"].is_array()) {
        for (const auto & msg : body["messages"]) {
            const std::string role = msg.value("role", std::string());
            if (!msg.contains("content")) {
                if (role == "assistant") continue;
                messages.push_back(msg);
                continue;
            }
            const auto & content = msg["content"];
            if (content.is_string() || !content.is_array()) {
                messages.push_back(msg);
                continue;
            }
            nlohmann::ordered_json tool_calls = nlohmann::ordered_json::array();
            nlohmann::ordered_json converted = nlohmann::ordered_json::array();
            nlohmann::ordered_json tool_results = nlohmann::ordered_json::array();
            std::string reasoning;
            bool has_tool_calls = false;
            for (const auto & block : content) {
                const std::string type = block.value("type", std::string());
                if (type == "text") {
                    converted.push_back(block);
                } else if (type == "thinking") {
                    reasoning += block.value("thinking", std::string());
                } else if (type == "image") {
                    const auto source = block.value("source", nlohmann::ordered_json::object());
                    const std::string stype = source.value("type", std::string());
                    if (stype == "base64") {
                        const std::string media = source.value("media_type", std::string("image/jpeg"));
                        converted.push_back({{"type", "image_url"},
                                             {"image_url", {{"url", "data:" + media + ";base64," + source.value("data", std::string())}}}});
                    } else if (stype == "url") {
                        converted.push_back({{"type", "image_url"},
                                             {"image_url", {{"url", source.value("url", std::string())}}}});
                    }
                } else if (type == "tool_use") {
                    tool_calls.push_back({{"id", block.value("id", std::string())},
                                          {"type", "function"},
                                          {"function", {{"name", block.value("name", std::string())},
                                                        {"arguments", block.value("input", nlohmann::ordered_json::object()).dump()}}}});
                    has_tool_calls = true;
                } else if (type == "tool_result") {
                    const auto rc = block.value("content", nlohmann::ordered_json());
                    std::string text;
                    if (rc.is_string()) text = rc.get<std::string>();
                    else if (rc.is_array()) {
                        for (const auto & c : rc) {
                            if (c.value("type", std::string()) == "text") text += c.value("text", std::string());
                        }
                    }
                    tool_results.push_back({{"role", "tool"},
                                            {"tool_call_id", block.value("tool_use_id", std::string())},
                                            {"content", text}});
                }
            }
            if (!converted.empty() || has_tool_calls || !reasoning.empty()) {
                nlohmann::ordered_json nm = {{"role", role}};
                if (!converted.empty()) nm["content"] = converted;
                else nm["content"] = "";
                if (!tool_calls.empty()) nm["tool_calls"] = tool_calls;
                if (!reasoning.empty()) nm["reasoning_content"] = reasoning;
                messages.push_back(std::move(nm));
            }
            for (const auto & tr : tool_results) messages.push_back(tr);
        }
    }
    oai["messages"] = std::move(messages);
    if (body.contains("tools") && body["tools"].is_array()) {
        nlohmann::ordered_json tools = nlohmann::ordered_json::array();
        for (const auto & tool : body["tools"]) {
            tools.push_back({{"type", "function"},
                             {"function", {{"name", tool.value("name", std::string())},
                                           {"description", tool.value("description", std::string())},
                                           {"parameters", tool.contains("input_schema") ? tool["input_schema"] : nlohmann::ordered_json::object()}}}});
        }
        oai["tools"] = std::move(tools);
    }
    if (body.contains("tool_choice") && body["tool_choice"].is_object()) {
        const std::string type = body["tool_choice"].value("type", std::string());
        if (type == "any" || type == "tool") oai["tool_choice"] = "required";
        else oai["tool_choice"] = type;
    }
    if (body.contains("stop_sequences")) oai["stop"] = body["stop_sequences"];
    oai["max_tokens"] = body.value("max_tokens", 4096);
    for (const auto & key : {"temperature", "top_p", "top_k", "stream"}) {
        if (body.contains(key)) oai[key] = body[key];
    }
    if (body.contains("thinking") && body["thinking"].is_object()) {
        if (body["thinking"].value("type", std::string()) == "enabled") {
            oai["thinking_budget_tokens"] = body["thinking"].value("budget_tokens", 10000);
        }
    }
    return oai;
}

static nlohmann::ordered_json openai_usage(const ServeResult & r) {
    return {{"prompt_tokens", r.prompt_tokens},
            {"completion_tokens", r.completion_tokens},
            {"total_tokens", r.prompt_tokens + r.completion_tokens}};
}

static nlohmann::ordered_json openai_message(const common_chat_msg & msg) {
    nlohmann::ordered_json m;
    m["role"] = "assistant";
    if (!msg.reasoning_content.empty()) m["reasoning_content"] = msg.reasoning_content;
    if (!msg.content.empty()) m["content"] = msg.content;
    else m["content"] = "";
    if (!msg.tool_calls.empty()) {
        nlohmann::ordered_json tcs = nlohmann::ordered_json::array();
        for (const auto & tc : msg.tool_calls) {
            tcs.push_back({{"id", tc.id}, {"type", "function"},
                           {"function", {{"name", tc.name}, {"arguments", tc.arguments}}}});
        }
        m["tool_calls"] = std::move(tcs);
    }
    return m;
}

static nlohmann::ordered_json openai_chunk_delta(const common_chat_msg_diff & d, bool & first_role) {
    nlohmann::ordered_json delta = nlohmann::ordered_json::object();
    if (first_role) {
        delta["role"] = "assistant";
        first_role = false;
    }
    if (!d.reasoning_content_delta.empty()) delta["reasoning_content"] = d.reasoning_content_delta;
    if (!d.content_delta.empty()) delta["content"] = d.content_delta;
    if (d.tool_call_index != std::string::npos) {
        nlohmann::ordered_json tc;
        tc["index"] = d.tool_call_index;
        if (!d.tool_call_delta.id.empty()) {
            tc["id"] = d.tool_call_delta.id;
            tc["type"] = "function";
        }
        if (!d.tool_call_delta.name.empty() || !d.tool_call_delta.arguments.empty()) {
            nlohmann::ordered_json fn = nlohmann::ordered_json::object();
            if (!d.tool_call_delta.name.empty()) fn["name"] = d.tool_call_delta.name;
            if (!d.tool_call_delta.arguments.empty()) fn["arguments"] = d.tool_call_delta.arguments;
            tc["function"] = fn;
        }
        delta["tool_calls"] = nlohmann::ordered_json::array({tc});
    }
    return delta;
}

static std::string openai_stream_chunk(const std::string & id, const std::string & model,
                                       const nlohmann::ordered_json & delta, const char * finish) {
    nlohmann::ordered_json chunk;
    chunk["id"] = id;
    chunk["object"] = "chat.completion.chunk";
    chunk["created"] = now_unix();
    chunk["model"] = model;
    nlohmann::ordered_json ch;
    ch["index"] = 0;
    ch["delta"] = delta;
    ch["finish_reason"] = finish ? nlohmann::ordered_json(finish) : nlohmann::ordered_json(nullptr);
    chunk["choices"] = nlohmann::ordered_json::array({ch});
    return "data: " + chunk.dump() + "\n\n";
}

static std::string anthropic_event(const char * ev, const nlohmann::ordered_json & data) {
    return std::string("event: ") + ev + "\ndata: " + data.dump() + "\n\n";
}

static std::string anthropic_stop_reason(const std::string & oai_reason) {
    if (oai_reason == "tool_calls") return "tool_use";
    if (oai_reason == "length") return "max_tokens";
    if (oai_reason == "stop") return "end_turn";
    return "end_turn";
}

static nlohmann::ordered_json anthropic_message(const ServeResult & r, const std::string & id, const std::string & model) {
    nlohmann::ordered_json msg;
    msg["id"] = id;
    msg["type"] = "message";
    msg["role"] = "assistant";
    nlohmann::ordered_json content = nlohmann::ordered_json::array();
    if (!r.msg.reasoning_content.empty()) {
        content.push_back({{"type", "thinking"}, {"thinking", r.msg.reasoning_content}});
    }
    if (!r.msg.content.empty()) {
        content.push_back({{"type", "text"}, {"text", r.msg.content}});
    }
    for (const auto & tc : r.msg.tool_calls) {
        nlohmann::ordered_json input;
        try {
            input = nlohmann::ordered_json::parse(tc.arguments);
        } catch (...) {
            input = tc.arguments;
        }
        content.push_back({{"type", "tool_use"}, {"id", tc.id}, {"name", tc.name}, {"input", input}});
    }
    msg["content"] = std::move(content);
    msg["model"] = model;
    msg["stop_reason"] = anthropic_stop_reason(r.finish_reason);
    msg["stop_sequence"] = nullptr;
    msg["usage"] = {{"input_tokens", r.prompt_tokens}, {"output_tokens", r.completion_tokens}};
    return msg;
}
static int cmd_bench(CliArgs & cli) {
    for (size_t i = 0; i < cli.sub_args.size(); i++) {
        const std::string & a = cli.sub_args[i];
        if ((a == "--ctx" || a == "-c") && i + 1 < cli.sub_args.size()) {
            parse_int(cli.sub_args[++i], cli.bench_ctx);
        } else if (a == "--tokens" && i + 1 < cli.sub_args.size()) {
            parse_int(cli.sub_args[++i], cli.bench_tokens);
        } else if (a == "--prompt" && i + 1 < cli.sub_args.size()) {
            cli.prompt = cli.sub_args[++i];
        } else if (a == "--ngl" || a == "--n-gpu-layers") {
            if (i + 1 < cli.sub_args.size()) { parse_int(cli.sub_args[++i], cli.ngl); }
        } else if (!a.empty() && a[0] != '-' && cli.model.empty()) {
            cli.model = a;
        } else {
            fprintf(stderr, "bench: unknown option '%s'\n", a.c_str());
            return 1;
        }
    }
    if (cli.model.empty()) {
        fprintf(stderr, "usage: anvil bench <model> [--ctx <n>] [--tokens <n>] [--prompt <text>]\n");
        return 1;
    }
    std::string path, friendly;
    if (!resolve_model_arg(cli.model, path, friendly)) {
        fprintf(stderr, "\033[31merror: '%s' is not a file and not a registered model\033[0m\n",
                cli.model.c_str());
        return 1;
    }
    if (!validate_gguf(path)) {
        fprintf(stderr, "\033[31merror: %s\033[0m\n", gguf_check_error(path).c_str());
        return 1;
    }
    cli.model = path;

    const HWInfo hw = probe_hw();
    const ModelMeta meta = read_model_meta(path);
    const int max_ctx = static_cast<int>(meta.trained_ctx);
    AnvilConfig base = config_exists() ? load_config() : AnvilConfig{};
    if (cli.ngl >= 0) base.ngl = cli.ngl;
    else if (base.ngl < 0) base.ngl = derive_ngl(hw);
    int ctx_n = cli.bench_ctx > 0 ? cli.bench_ctx : (max_ctx > 0 ? max_ctx : 2048);

    LlamaBackend backend;
    fprintf(stderr, "Loading model: %s ...\n", path.c_str());
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = base.ngl;
    LlamaModel model(llama_model_load_from_file(path.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "\033[31merror: failed to load model '%s'\033[0m\n", path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const std::string prompt = cli.prompt.empty()
        ? "The quick brown fox jumps over the lazy dog. What does that say about"
        : cli.prompt;
    std::vector<llama_token> prompt_toks = tokenize_render(vocab, prompt);
    if (prompt_toks.empty()) {
        fprintf(stderr, "\033[31merror: prompt tokenization failed\033[0m\n");
        return 1;
    }

    const int n_tokens = cli.bench_tokens > 0 ? cli.bench_tokens : 32;
    struct Pair { const char * name; ggml_type k; ggml_type v; };
    const Pair pairs[] = {
        { "f16/f16        ", GGML_TYPE_F16,       GGML_TYPE_F16      },
        { "q8_0/turbo3    ", GGML_TYPE_Q8_0,      GGML_TYPE_TURBO3_0 },
        { "turbo4/turbo3  ", GGML_TYPE_TURBO4_0,  GGML_TYPE_TURBO3_0 },
        { "turbo4/turbo2  ", GGML_TYPE_TURBO4_0,  GGML_TYPE_TURBO2_0 },
    };
    const int n_pairs = static_cast<int>(sizeof(pairs) / sizeof(pairs[0]));

    printf("\n\033[1;36m── TurboQuant KV Benchmark ──\033[0m\n");
    printf("  model   : %s\n", path.c_str());
    printf("  ctx     : %d tokens\n", ctx_n);
    printf("  prompt  : %zu tokens | generate %d\n\n", prompt_toks.size(), n_tokens);
    printf("%-18s %-12s %-10s %-8s %s\n", "KV CONFIG", "TOK/S", "TOKENS", "FIRST", "RESULT");

    for (int p = 0; p < n_pairs; p++) {
        AnvilConfig cfg = base;
        cfg.type_k = pairs[p].k;
        cfg.type_v = pairs[p].v;
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = static_cast<uint32_t>(ctx_n);
        cparams.n_batch = static_cast<uint32_t>(ctx_n);
        cparams.n_threads = cfg.n_threads > 0 ? cfg.n_threads : hw.cpu_threads;
        cparams.flash_attn_type = cfg.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cparams.type_k = cfg.type_k;
        cparams.type_v = cfg.type_v;
        const auto t0 = std::chrono::steady_clock::now();
        LlamaContext ctx(llama_init_from_model(model, cparams));
        if (!ctx) {
            printf("%-18s %-12s %-10s %-8s \033[31mcontext creation failed\033[0m\n",
                   pairs[p].name, "-", "-", "-");
            continue;
        }
        const double ctx_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        llama_memory_t mem = llama_get_memory(ctx);
        LlamaSampler smpl(build_sampler_chain(vocab, cfg, false, ""));
        if (!smpl) {
            printf("%-18s %-12s %-10s %-8s \033[31msampler failed\033[0m\n",
                   pairs[p].name, "-", "-", "-");
            continue;
        }
        std::vector<llama_token> full = prompt_toks;
        const llama_token bos = llama_vocab_bos(vocab);
        if (llama_vocab_get_add_bos(vocab) && (full.empty() || full[0] != bos)) {
            full.insert(full.begin(), bos);
        }
        bool ok = true;
        const int32_t chunk = static_cast<int32_t>(llama_n_batch(ctx));
        for (size_t i = 0; i < full.size(); i += static_cast<size_t>(chunk)) {
            const size_t nn = std::min<size_t>(static_cast<size_t>(chunk), full.size() - i);
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(full.data() + i),
                                                    static_cast<int32_t>(nn));
            if (llama_decode(ctx, batch) != 0) { ok = false; break; }
        }
        if (!ok) {
            printf("%-18s %-12s %-10s %-8s \033[31mdecode failed\033[0m\n",
                   pairs[p].name, "-", "-", "-");
            continue;
        }
        llama_sampler_reset(smpl.get());
        const auto g0 = std::chrono::steady_clock::now();
        int gen = 0;
        std::atomic<bool> no_stop{false};
        while (gen < n_tokens) {
            const llama_token id = llama_sampler_sample(smpl.get(), ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&id), 1);
            if (llama_decode(ctx, batch) != 0) break;
            gen++;
        }
        const double gen_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - g0).count();
        const double tps = gen_sec > 0 ? gen / gen_sec : 0.0;
        printf("%-18s %-12.1f %-10d %-8.2f %s\n",
               pairs[p].name, tps, gen, ctx_sec,
               gen >= n_tokens ? "\033[32mok\033[0m" : "\033[33mearly stop\033[0m");
        (void)no_stop;
    }
    printf("\n");
    return 0;
}

static void doctor_check(bool ok, const char * name, const std::string & detail) {
    printf("  %s %s%s\n", ok ? "\033[32m[ OK ]\033[0m" : "\033[31m[FAIL]\033[0m",
           name, detail.empty() ? "" : ("  " + detail).c_str());
}

static int cmd_doctor(const std::vector<std::string> & args) {
    (void)args;
    printf("\n\033[1;36m── anvil doctor ──\033[0m\n");
    int failures = 0;
    auto check = [&](bool ok, const char * name, const std::string & detail = "") {
        doctor_check(ok, name, detail);
        if (!ok) failures++;
    };

    const HWInfo hw = probe_hw();
    std::string hwline = hw.cpu;
    if (!hw.arch.empty()) hwline += " | " + hw.arch;
    check(!hw.cpu.empty(), "hardware probe", hwline);
    check(hw.ram_bytes > 0, "memory detect",
          std::to_string(hw.ram_bytes / (1024ULL * 1024 * 1024)) + " GB");
    check(hw.cpu_threads > 0, "cpu threads", std::to_string(hw.cpu_threads));
    check(!hw.gpus.empty() || hw.apple_silicon, "gpu detect",
          hw.gpus.empty() ? (hw.apple_silicon ? "Apple GPU" : "CPU only") : hw.gpus[0].name);

    check(config_exists(), "config.json exists", config_path());
    AnvilConfig cfg = load_config();
    check(cfg.version == CONFIG_VERSION, "config.json version",
          "v" + std::to_string(cfg.version));

    std::vector<ModelEntry> models = load_models();
    check(true, "models.json readable",
          std::to_string(models.size()) + " registered model(s)");
    int bad = 0;
    uint64_t total_bytes = 0;
    for (const auto & m : models) {
        const bool exists = std::filesystem::exists(m.path);
        const bool gguf = exists && validate_gguf(m.path);
        if (!exists || !gguf) bad++;
        std::error_code ec;
        const auto sz = std::filesystem::file_size(m.path, ec);
        if (!ec) total_bytes += sz;
    }
    check(bad == 0, "model files present + valid GGUF",
          std::to_string(models.size() - static_cast<size_t>(bad)) + "/" + std::to_string(models.size()) + " ok");

    std::error_code spc;
    const auto space = std::filesystem::space(config_dir(), spc);
    check(!spc, "disk space",
          format_size(space.available) + " free in " + config_dir());

    {
        const std::string probe = capture("which curl 2>/dev/null || which wget 2>/dev/null");
        check(!probe.empty(), "download tool (curl/wget)");
    }
    {
        const std::string body = http_get("https://huggingface.co/api/models/Qwen/Qwen3.6-27B", "--max-time 10");
        check(!body.empty(), "network reachability (huggingface.co)");
    }
    check(llama_supports_gpu_offload() || hw.apple_silicon, "gpu offload backend",
          llama_supports_gpu_offload() ? "available" : "CPU only");

    printf("\n  %s\n\n", failures == 0
        ? "\033[1;32mAll checks passed.\033[0m"
        : (std::string("\033[1;31m") + std::to_string(failures) + " check(s) failed.\033[0m").c_str());
    return failures == 0 ? 0 : 1;
}

static std::string current_exe_path() {
    char buf[4096];
#ifdef _WIN32
    const DWORD n = GetModuleFileNameA(nullptr, buf, sizeof(buf));
    if (n > 0 && n < sizeof(buf)) return std::string(buf, n);
#elif defined(__APPLE__)
    uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) == 0) return std::string(buf);
#else
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) {
        buf[n] = '\0';
        return std::string(buf);
    }
#endif
    return "";
}

static std::string asset_name_for_platform(const std::string & os, const std::string & arch) {
    if (os == "linux" && arch == "x86_64")   return "anvil-linux-x86_64";
    if (os == "linux" && arch == "aarch64")  return "anvil-linux-aarch64";
    if (os == "macos" && arch == "aarch64")  return "anvil-macos-aarch64";
    if (os == "macos" && arch == "x86_64")   return "anvil-macos-x86_64";
    if (os == "windows")                     return "anvil-windows-x86_64.exe";
    return "";
}

static int cmd_self_update(const std::vector<std::string> & args) {
    (void)args;
    const std::string url = "https://api.github.com/repos/Anvil-LLM/anvil/releases/latest";
    const std::string body = http_get(url);
    if (body.empty()) {
        fprintf(stderr, "\033[31merror: could not reach GitHub releases API\033[0m\n");
        return 1;
    }
    nlohmann::json j;
    try {
        j = nlohmann::json::parse(body);
    } catch (const nlohmann::json::exception &) {
        fprintf(stderr, "\033[31merror: invalid GitHub API response\033[0m\n");
        return 1;
    }
    if (!j.contains("tag_name")) {
        fprintf(stderr, "\033[31merror: no latest release found\033[0m\n");
        return 1;
    }
    const std::string tag = j["tag_name"].get<std::string>();
    const std::string want = asset_name_for_platform(probe_hw().os, probe_hw().arch);
    if (want.empty()) {
        fprintf(stderr, "\033[31merror: no prebuilt binary for this platform\033[0m\n");
        return 1;
    }
    if (tag == std::string("v") + ANVIL_VERSION) {
        printf("Already up to date (%s).\n", ANVIL_VERSION);
        return 0;
    }
    if (!j.contains("assets") || !j["assets"].is_array()) {
        fprintf(stderr, "\033[31merror: release has no assets\033[0m\n");
        return 1;
    }
    std::string asset_url, digest;
    for (const auto & a : j["assets"]) {
        if (!a.is_object() || !a.contains("name")) continue;
        const std::string nm = a["name"].get<std::string>();
        if (nm != want) continue;
        if (a.contains("browser_download_url")) asset_url = a["browser_download_url"].get<std::string>();
        if (a.contains("digest") && a["digest"].is_string()) {
            std::string d = a["digest"].get<std::string>();
            if (d.rfind("sha256:", 0) == 0) digest = d.substr(7);
        }
        break;
    }
    if (asset_url.empty()) {
        fprintf(stderr, "\033[31merror: asset '%s' not found in release %s\033[0m\n",
                want.c_str(), tag.c_str());
        return 1;
    }
    const std::string exe = current_exe_path();
    if (exe.empty()) {
        fprintf(stderr, "\033[31merror: could not determine current binary path\033[0m\n");
        return 1;
    }
    printf("Updating %s -> %s\n", ANVIL_VERSION, tag.c_str());
    printf("  downloading %s ...\n", want.c_str());
    const std::string tmp = exe + ".new";
    std::error_code ec;
    std::filesystem::remove(tmp, ec);
    if (http_download(asset_url, tmp) != 0) {
        fprintf(stderr, "\033[31merror: download failed\033[0m\n");
        return 1;
    }
    if (!digest.empty()) {
        const std::string actual = sha256_of(tmp);
        if (actual.empty()) {
            fprintf(stderr, "\033[33mwarning: no sha256 tool; skipping verification\033[0m\n");
        } else if (actual != digest) {
            fprintf(stderr, "\033[31merror: checksum mismatch (got %s, expected %s)\033[0m\n",
                    actual.c_str(), digest.c_str());
            std::filesystem::remove(tmp, ec);
            return 1;
        } else {
            printf("  verified sha256:%s\n", digest.c_str());
        }
    } else {
        fprintf(stderr, "\033[33mwarning: no digest in release metadata; skipping verification\033[0m\n");
    }
#ifdef _WIN32
    std::filesystem::rename(tmp, exe, ec);
#else
    std::filesystem::permissions(tmp, std::filesystem::perms::owner_all |
                                        std::filesystem::perms::group_read |
                                        std::filesystem::perms::group_exec |
                                        std::filesystem::perms::others_read |
                                        std::filesystem::perms::others_exec, ec);
    if (!ec) std::filesystem::rename(tmp, exe, ec);
#endif
    if (ec) {
        fprintf(stderr, "\033[31merror: could not replace binary (%s). New binary kept at %s\033[0m\n",
                ec.message().c_str(), tmp.c_str());
        return 1;
    }
    printf("\033[32mUpdated to %s. Restart to use the new binary.\033[0m\n", tag.c_str());
    return 0;
}


struct RagChunk {
    std::string text;
    std::vector<std::string> toks;
    size_t start = 0;
    size_t end = 0;
};

struct RagIndex {
    std::vector<RagChunk> chunks;
    std::map<std::string, double> idf;
    std::map<std::string, std::vector<size_t>> postings;
    std::vector<std::string> sources;
    bool empty() const { return chunks.empty(); }
    size_t size() const { return chunks.size(); }

    void rebuild_idf() {
        idf.clear();
        if (chunks.empty()) return;
        for (const auto & [t, ps] : postings) {
            const double df = static_cast<double>(ps.size());
            idf[t] = std::log(static_cast<double>(chunks.size()) / (1.0 + df)) + 1.0;
        }
    }

    bool add_source(const std::string & text, const std::string & source) {
        const size_t before = chunks.size();
        add_text(text);
        if (chunks.size() == before) return false;
        sources.push_back(source);
        rebuild_idf();
        return true;
    }

    bool add_file(const std::string & path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (content.size() > (8 << 20)) content.resize(8 << 20);
        return add_source(content, path);
    }

    bool add_url(const std::string & url) {
        httplib::Client cli(url);
        cli.set_follow_location(true);
        cli.set_connection_timeout(15, 0);
        cli.set_read_timeout(30, 0);
        auto res = cli.Get("/");
        if (!res || res->status != 200) return false;
        std::string body = res->body;
        std::string stripped;
        bool in_tag = false;
        for (const char c : body) {
            if (c == '<') { in_tag = true; continue; }
            if (c == '>') { in_tag = false; stripped += ' '; continue; }
            if (!in_tag) stripped += c;
        }
        std::string clean;
        bool prev_space = false;
        for (const char c : stripped) {
            if (c == '\r' || c == '\t') continue;
            if (c == '\n' || c == ' ') {
                if (prev_space) continue;
                prev_space = true;
                clean += ' ';
            } else {
                prev_space = false;
                clean += c;
            }
        }
        return add_source(clean, url);
    }

    static std::vector<std::string> tokenize(const std::string & s) {
        std::vector<std::string> out;
        std::string cur;
        for (const unsigned char c : s) {
            if (std::isalnum(c)) {
                cur += static_cast<char>(std::tolower(c));
            } else if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        }
        if (!cur.empty()) out.push_back(cur);
        return out;
    }

    void add_text(const std::string & text) {
        std::vector<std::string> toks = tokenize(text);
        if (toks.size() < 8) return;
        const size_t CHUNK = 300;
        for (size_t i = 0; i < toks.size(); i += CHUNK) {
            const size_t n = std::min(CHUNK, toks.size() - i);
            RagChunk c;
            c.toks.assign(toks.begin() + static_cast<long>(i), toks.begin() + static_cast<long>(i + n));
            std::set<std::string> uniq(c.toks.begin(), c.toks.end());
            for (const auto & t : uniq) postings[t].push_back(chunks.size());
            c.text = join_tokens(c.toks);
            chunks.push_back(std::move(c));
        }
    }

    static std::string join_tokens(const std::vector<std::string> & toks) {
        std::string out;
        for (size_t i = 0; i < toks.size(); i++) {
            if (i > 0) out += ' ';
            out += toks[i];
        }
        return out;
    }

    bool build_dir(const std::string & dir) {
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) return false;
        std::vector<std::string> files;
        for (const auto & de : std::filesystem::recursive_directory_iterator(dir, ec)) {
            if (!de.is_regular_file(ec)) continue;
            const std::string ext = de.path().extension().string();
            static const char * ok[] = {".txt", ".md", ".rst", ".json", ".yaml", ".yml",
                                        ".py", ".cpp", ".c", ".h", ".hpp", ".rs", ".go",
                                        ".js", ".ts", ".sh", ".toml", ".ini", ".csv", ".log"};
            bool good = false;
            for (const char * e : ok) {
                if (ext == e) { good = true; break; }
            }
            if (good) files.push_back(de.path().string());
        }
        std::sort(files.begin(), files.end());
        for (const auto & f : files) {
            std::ifstream in(f, std::ios::binary);
            if (!in) continue;
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            if (content.size() > (8 << 20)) content.resize(8 << 20);
            const size_t before = chunks.size();
            add_text(content);
            if (chunks.size() > before) sources.push_back(f);
        }
        rebuild_idf();
        return !chunks.empty();
    }

    std::vector<size_t> search(const std::string & query, size_t k) const {
        const std::vector<std::string> qtoks = tokenize(query);
        std::map<size_t, double> scores;
        for (const auto & t : qtoks) {
            const auto it = postings.find(t);
            if (it == postings.end()) continue;
            const auto itf = idf.find(t);
            const double w = itf != idf.end() ? itf->second : 0.0;
            for (const size_t ci : it->second) scores[ci] += w;
        }
        std::vector<std::pair<double, size_t>> ranked;
        for (const auto & [ci, sc] : scores) ranked.push_back({sc, ci});
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto & a, const auto & b) { return a.first > b.first; });
        std::vector<size_t> out;
        const size_t n = std::min(k, ranked.size());
        for (size_t i = 0; i < n; i++) out.push_back(ranked[i].second);
        return out;
    }

    std::string context_for(const std::string & query, size_t k) const {
        const std::vector<size_t> hits = search(query, k);
        if (hits.empty()) return "";
        std::string out;
        for (const size_t ci : hits) {
            out += chunks[ci].text;
            out += "\n\n";
        }
        return out;
    }
};

struct VisionSession {
    mtmd_context * ctx = nullptr;
    bool ok = false;

    ~VisionSession() {
        if (ctx) mtmd_free(ctx);
    }

    bool init(const std::string & mmproj, llama_model * model, const HWInfo & hw, int n_threads) {
        if (ctx) return ok;
        mtmd_context_params p = mtmd_context_params_default();
        p.use_gpu = hw.apple_silicon || !hw.gpus.empty();
        p.n_threads = n_threads > 0 ? n_threads : hw.cpu_threads;
        p.warmup = true;
        ctx = mtmd_init_from_file(mmproj.c_str(), model, p);
        if (!ctx) return false;
        ok = mtmd_support_vision(ctx);
        return ok;
    }

    bool eval_image_turn(llama_context * lctx, const std::string & formatted,
                         const std::string & image_path, llama_pos * new_past,
                         int32_t n_batch) {
        if (!ctx) return false;
        mtmd_input_chunks * chunks = mtmd_input_chunks_init();
        if (!chunks) return false;
        const auto bw = mtmd_helper_bitmap_init_from_file(ctx, image_path.c_str(), false);
        if (!bw.bitmap) {
            mtmd_input_chunks_free(chunks);
            return false;
        }
        std::string prompt = formatted;
        const char * marker = mtmd_get_marker(ctx);
        if (marker && prompt.find(marker) == std::string::npos) {
            prompt += "\n";
            prompt += marker;
            prompt += "\n";
        }
        mtmd_input_text text;
        text.text = prompt.c_str();
        text.add_special = true;
        text.parse_special = true;
        const mtmd_bitmap * bm = bw.bitmap;
        const int32_t rc = mtmd_tokenize(ctx, chunks, &text, &bm, 1);
        if (rc != 0) {
            mtmd_bitmap_free(bw.bitmap);
            mtmd_input_chunks_free(chunks);
            return false;
        }
        const int32_t erc = mtmd_helper_eval_chunks(ctx, lctx, chunks, 0, 0, n_batch, true, new_past);
        mtmd_bitmap_free(bw.bitmap);
        mtmd_input_chunks_free(chunks);
        return erc == 0;
    }
};

static std::string resolve_system_prompt(const std::string & sp) {
    if (sp.size() > 1 && sp[0] == '@') {
        const std::string preset = load_preset_text(sp);
        if (preset.empty()) {
            fprintf(stderr, "\033[33mwarning: preset '%s' not found in %s (available: ",
                    sp.c_str(), presets_dir().c_str());
            const auto names = list_presets();
            for (size_t i = 0; i < names.size(); i++) {
                fprintf(stderr, "%s%s", i > 0 ? ", " : "", names[i].c_str());
            }
            fprintf(stderr, ")\033[0m\n");
            return "";
        }
        return preset;
    }
    return sp;
}
static bool maybe_auto_pull(std::string & model, std::string & path, std::string & friendly) {
    if (model.rfind("hf:", 0) == 0 || model.rfind("ollama:", 0) == 0) {
        const std::vector<std::string> pa = { model };
        if (cmd_pull(pa) != 0) return false;
        std::string stripped = model.substr(model.find(':') + 1);
        if (model.rfind("ollama:", 0) == 0) {
            const size_t colon = stripped.find(':');
            if (colon != std::string::npos) stripped = stripped.substr(0, colon);
            const size_t slash = stripped.find('/');
            if (slash != std::string::npos) stripped = stripped.substr(slash + 1);
            model = slugify(stripped + "-latest");
        } else {
            const std::vector<ModelEntry> models = load_models();
            if (!models.empty()) model = models.back().name;
        }
        return resolve_model_arg(model, path, friendly);
    }
    return false;
}
struct McpServerCfg {
    std::string name;
    std::string command;
    std::vector<std::string> args;
    bool enabled = true;

    nlohmann::json to_json() const {
        nlohmann::json j;
        j["name"] = name;
        j["command"] = command;
        j["args"] = args;
        j["enabled"] = enabled;
        return j;
    }

    static McpServerCfg from_json(const nlohmann::json & j) {
        McpServerCfg c;
        if (j.contains("name") && j["name"].is_string()) c.name = j["name"].get<std::string>();
        if (j.contains("command") && j["command"].is_string()) c.command = j["command"].get<std::string>();
        if (j.contains("args") && j["args"].is_array()) {
            for (const auto & a : j["args"]) {
                if (a.is_string()) c.args.push_back(a.get<std::string>());
            }
        }
        if (j.contains("enabled") && j["enabled"].is_boolean()) c.enabled = j["enabled"].get<bool>();
        return c;
    }
};

inline std::string mcp_json_path() { return config_dir() + "/mcp.json"; }

static std::vector<McpServerCfg> load_mcp_servers() {
    std::vector<McpServerCfg> out;
    std::ifstream f(mcp_json_path());
    if (!f) return out;
    nlohmann::json root;
    try {
        f >> root;
        if (root.contains("servers") && root["servers"].is_array()) {
            for (const auto & j : root["servers"]) {
                try { out.push_back(McpServerCfg::from_json(j)); }
                catch (const nlohmann::json::exception &) {}
            }
        }
    } catch (const nlohmann::json::exception & e) {
        fprintf(stderr, "\033[33mwarning: %s unreadable (%s)\033[0m\n",
                mcp_json_path().c_str(), e.what());
    }
    return out;
}

static bool save_mcp_servers(const std::vector<McpServerCfg> & servers) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(config_dir(), ec);
    nlohmann::json root;
    root["version"] = 1;
    root["servers"] = nlohmann::json::array();
    for (const auto & s : servers) root["servers"].push_back(s.to_json());
    const std::string tmp = mcp_json_path() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return false;
        f << root.dump(2) << "\n";
        f.flush();
        if (!f) return false;
    }
    fs::rename(tmp, mcp_json_path(), ec);
    return !ec;
}

struct McpTool {
    std::string server;
    std::string name;
    std::string description;
    nlohmann::json input_schema;
};

static std::string mcp_tool_full_name(const McpTool & t) {
    return "mcp__" + t.server + "__" + t.name;
}

struct McpClient {
    std::string name;
    std::string error;
    std::vector<McpTool> tools;
    std::unique_ptr<mcp::client::Client> client;

    bool open_stdio(const std::string & command, const std::vector<std::string> & args) {
        mcp::client::Client::StdioEndpoint ep;
        ep.command = command;
        ep.args = args;
        try {
            client = std::make_unique<mcp::client::Client>(
                mcp::client::Client::connect_stdio(std::move(ep)));
        } catch (const std::exception & e) {
            error = e.what();
            return false;
        }
        return client != nullptr;
    }

    bool open_http(const std::string & url) {
        try {
            client = std::make_unique<mcp::client::Client>(
                mcp::client::Client::connect_streamable_http(url));
        } catch (const std::exception & e) {
            error = e.what();
            return false;
        }
        return client != nullptr;
    }

    bool start() {
        auto started = client->start();
        if (!started) { error = started.error().message; return false; }
        auto init = client->initialize("anvil", ANVIL_VERSION);
        if (!init) { error = init.error().message; return false; }
        auto notif = client->notify_initialized();
        if (!notif) { error = notif.error().message; return false; }
        auto listed = client->list_all_tools();
        if (!listed) { error = listed.error().message; return false; }
        for (const auto & t : *listed) {
            McpTool mt;
            mt.server = name;
            mt.name = t.name;
            mt.description = t.description;
            mt.input_schema = t.input_schema;
            tools.push_back(std::move(mt));
        }
        return true;
    }

    bool call(const std::string & tool, const nlohmann::json & args, std::string & out) {
        auto res = client->call_raw(tool, args);
        if (!res) {
            out = "error: " + res.error().message;
            return false;
        }
        std::string text;
        for (const auto & block : res->content) {
            if (block.type == "text" || block.type.empty()) text += block.text;
            else text += "[" + block.type + " result]";
        }
        if (text.empty() && res->structured_content.has_value()) text = res->structured_content->dump();
        if (text.empty()) text = "(empty result)";
        out = std::move(text);
        return true;
    }

    void stop() {
        if (client) {
            try { client->stop(); } catch (...) {}
        }
    }
};

struct McpHost {
    std::vector<McpServerCfg> servers;
    std::vector<McpClient> clients;

    size_t tool_count() const {
        size_t n = 0;
        for (const auto & c : clients) n += c.tools.size();
        return n;
    }

    void connect_all() {
        servers = load_mcp_servers();
        for (auto & s : servers) {
            if (!s.enabled) continue;
            McpClient c;
            c.name = s.name;
            bool ok = false;
            if (s.command.rfind("http://", 0) == 0 || s.command.rfind("https://", 0) == 0) {
                ok = c.open_http(s.command);
            } else {
                ok = c.open_stdio(s.command, s.args);
            }
            if (ok) ok = c.start();
            if (!ok) {
                fprintf(stderr, "  mcp %-16s \033[31mfailed: %s\033[0m\n", c.name.c_str(),
                        c.error.empty() ? "connect error" : c.error.c_str());
            }
            clients.push_back(std::move(c));
        }
    }

    std::vector<McpTool> all_tools() const {
        std::vector<McpTool> out;
        for (const auto & c : clients) {
            for (const auto & t : c.tools) out.push_back(t);
        }
        return out;
    }

    bool call(const std::string & full_name, const nlohmann::json & args, std::string & result) {
        for (auto & c : clients) {
            for (const auto & t : c.tools) {
                if (mcp_tool_full_name(t) == full_name) {
                    return c.call(t.name, args, result);
                }
            }
        }
        result = "error: MCP tool '" + full_name + "' not found";
        return false;
    }

    void shutdown() {
        for (auto & c : clients) c.stop();
    }
};

static nlohmann::ordered_json mcp_tools_to_oai(const std::vector<McpTool> & tools) {
    nlohmann::ordered_json arr = nlohmann::ordered_json::array();
    for (const auto & t : tools) {
        nlohmann::ordered_json fn;
        fn["type"] = "function";
        fn["function"]["name"] = mcp_tool_full_name(t);
        fn["function"]["description"] = t.description;
        if (t.input_schema.is_object() && !t.input_schema.empty()) {
            fn["function"]["parameters"] = t.input_schema;
        } else {
            fn["function"]["parameters"] = nlohmann::ordered_json{
                {"type", "object"}, {"properties", nlohmann::ordered_json::object()}};
        }
        arr.push_back(fn);
    }
    return arr;
}

static nlohmann::ordered_json chat_history_to_oai(const std::vector<ChatMessage> & history) {
    nlohmann::ordered_json arr = nlohmann::ordered_json::array();
    for (const auto & m : history) {
        if (m.role == "tool") {
            nlohmann::ordered_json j;
            j["role"] = "tool";
            j["tool_call_id"] = m.tool_call_id;
            j["content"] = m.content;
            arr.push_back(std::move(j));
        } else if (m.role == "assistant" && !m.tool_calls_json.empty()) {
            nlohmann::ordered_json j;
            j["role"] = "assistant";
            j["content"] = m.content;
            try { j["tool_calls"] = nlohmann::ordered_json::parse(m.tool_calls_json); }
            catch (...) { j["tool_calls"] = nlohmann::ordered_json::array(); }
            arr.push_back(std::move(j));
        } else {
            nlohmann::ordered_json j;
            j["role"] = m.role;
            j["content"] = m.content;
            arr.push_back(std::move(j));
        }
    }
    return arr;
}

static std::string chat_tool_calls_to_json(const std::vector<common_chat_tool_call> & tcs) {
    nlohmann::ordered_json arr = nlohmann::ordered_json::array();
    for (const auto & tc : tcs) {
        nlohmann::ordered_json j;
        j["id"] = tc.id;
        j["type"] = "function";
        j["function"]["name"] = tc.name;
        j["function"]["arguments"] = tc.arguments;
        arr.push_back(std::move(j));
    }
    return arr.dump();
}

inline std::string keys_path() { return config_dir() + "/keys.json"; }

struct RemoteKeys {
    std::string openai;
    std::string anthropic;
};

static RemoteKeys load_remote_keys() {
    RemoteKeys k;
    const char * o = std::getenv("OPENAI_API_KEY");
    const char * a = std::getenv("ANTHROPIC_API_KEY");
    if (o && o[0]) k.openai = o;
    if (a && a[0]) k.anthropic = a;
    std::ifstream f(keys_path());
    if (!f) return k;
    nlohmann::json j;
    try {
        f >> j;
        if (j.contains("openai") && j["openai"].is_string()) k.openai = j["openai"].get<std::string>();
        if (j.contains("anthropic") && j["anthropic"].is_string()) k.anthropic = j["anthropic"].get<std::string>();
    } catch (const nlohmann::json::exception &) {}
    return k;
}

static bool save_remote_keys(const RemoteKeys & k) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(config_dir(), ec);
    nlohmann::json j;
    if (!k.openai.empty()) j["openai"] = k.openai;
    if (!k.anthropic.empty()) j["anthropic"] = k.anthropic;
    const std::string tmp = keys_path() + ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (!f) return false;
        f << j.dump(2) << "\n";
        f.flush();
        if (!f) return false;
    }
    fs::permissions(tmp, fs::perms::owner_read | fs::perms::owner_write, ec);
    ec.clear();
    fs::rename(tmp, keys_path(), ec);
    return !ec;
}

static bool remote_chat(const std::string & provider, const std::string & model,
                        const nlohmann::ordered_json & messages,
                        const std::string & system_prompt, int max_tokens,
                        std::string & out_text,
                        const std::function<void(const std::string &)> & on_delta = {}) {
    const RemoteKeys keys = load_remote_keys();
    const std::string & key = provider == "anthropic" ? keys.anthropic : keys.openai;
    if (key.empty()) {
        out_text = "error: no " + provider + " API key. Set " +
                   (provider == "anthropic" ? "ANTHROPIC_API_KEY" : "OPENAI_API_KEY") +
                   " or run: anvil keys set " + provider + " <key>";
        return false;
    }
    nlohmann::ordered_json body;
    body["model"] = model;
    if (!system_prompt.empty()) {
        nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
        msgs.push_back({{{"role", "system"}, {"content", system_prompt}}});
        for (const auto & m : messages) msgs.push_back(m);
        body["messages"] = std::move(msgs);
    } else {
        body["messages"] = messages;
    }
    body["stream"] = true;
    body["max_tokens"] = max_tokens > 0 ? max_tokens : 2048;

    std::string url, path;
    httplib::Headers hdrs;
    const char * base_env = provider == "anthropic"
        ? std::getenv("ANVIL_ANTHROPIC_BASE_URL") : std::getenv("ANVIL_OPENAI_BASE_URL");
    if (provider == "anthropic") {
        url = base_env && base_env[0] ? base_env : "https://api.anthropic.com";
        path = "/v1/messages";
        hdrs = {{"x-api-key", key}, {"anthropic-version", "2023-06-01"},
               {"Content-Type", "application/json"}};
    } else {
        url = base_env && base_env[0] ? base_env : "https://api.openai.com";
        path = "/v1/chat/completions";
        hdrs = {{"Authorization", "Bearer " + key}, {"Content-Type", "application/json"}};
    }

    httplib::Client cli(url);
    cli.set_connection_timeout(30, 0);
    cli.set_read_timeout(300, 0);
    cli.set_follow_location(true);

    std::string sse_buf;
    std::string raw_body;
    auto on_chunk = [&](const char * data, size_t len) -> bool {
        raw_body.append(data, len);
        sse_buf.append(data, len);
        size_t pos = 0;
        while (true) {
            const size_t nl = sse_buf.find('\n', pos);
            if (nl == std::string::npos) break;
            const std::string line = sse_buf.substr(pos, nl - pos);
            pos = nl + 1;
            if (line.rfind("data:", 0) != 0) continue;
            const std::string payload = line.size() > 5 ? line.substr(5) : "";
            if (payload == "[DONE]") continue;
            try {
                const nlohmann::json ev = nlohmann::json::parse(payload);
                if (provider == "anthropic") {
                    if (ev.value("type", std::string()) == "content_block_delta") {
                        const auto & delta = ev["delta"];
                        if (delta.value("type", std::string()) == "text_delta") {
                            const std::string txt = delta.value("text", std::string());
                            out_text += txt;
                            if (on_delta) on_delta(txt);
                        }
                    }
                } else if (ev.contains("choices") && ev["choices"].is_array() &&
                           !ev["choices"].empty()) {
                    const auto & ch = ev["choices"][0];
                    if (ch.contains("delta")) {
                        const std::string txt = ch["delta"].value("content", std::string());
                        if (!txt.empty()) {
                            out_text += txt;
                            if (on_delta) on_delta(txt);
                        }
                    }
                }
            } catch (const nlohmann::json::exception &) {}
        }
        sse_buf.erase(0, pos);
        return true;
    };
    const std::string body_str = body.dump();
    auto res = cli.Post(path, hdrs,
        [&](size_t offset, httplib::DataSink & sink) -> bool {
            if (offset < body_str.size()) {
                sink.write(body_str.data() + offset, body_str.size() - offset);
            } else {
                sink.done();
            }
            return true;
        },
        "application/json", on_chunk);
    if (!res) {
        out_text = "error: request to " + url + path + " failed: " + httplib::to_string(res.error());
        return false;
    }
    if (res->status != 200) {
        std::string detail = raw_body.empty() ? res->body : raw_body;
        if (detail.size() > 400) detail.resize(400);
        out_text = "error: HTTP " + std::to_string(res->status) + " from " + provider + ": " + detail;
        return false;
    }
    if (out_text.empty()) {
        out_text = "(empty response)";
        return false;
    }
    return true;
}

static int cmd_keys(const std::vector<std::string> & args) {
    if (args.empty() || args[0] == "list") {
        const RemoteKeys k = load_remote_keys();
        auto mask = [](const std::string & s) {
            if (s.size() < 8) return std::string();
            return s.substr(0, 4) + "\xe2\x80\xa6" + s.substr(s.size() - 4);
        };
        printf("\n\033[1;36m── API Keys ──\033[0m\n");
        printf("  openai    : %s\n", k.openai.empty() ? "(not set)" : mask(k.openai).c_str());
        printf("  anthropic : %s\n", k.anthropic.empty() ? "(not set)" : mask(k.anthropic).c_str());
        printf("\nusage: anvil keys set <openai|anthropic> <key>\n");
        printf("       anvil keys unset <openai|anthropic>\n\n");
        return 0;
    }
    if (args[0] == "set" && args.size() >= 3) {
        if (args[1] != "openai" && args[1] != "anthropic") {
            fprintf(stderr, "error: provider must be 'openai' or 'anthropic'\n");
            return 1;
        }
        RemoteKeys k = load_remote_keys();
        if (args[1] == "openai") k.openai = args[2];
        else k.anthropic = args[2];
        if (!save_remote_keys(k)) {
            fprintf(stderr, "\033[31merror: could not save %s\033[0m\n", keys_path().c_str());
            return 1;
        }
        printf("Saved %s key to %s\n", args[1].c_str(), keys_path().c_str());
        return 0;
    }
    if (args[0] == "unset" && args.size() >= 2) {
        if (args[1] != "openai" && args[1] != "anthropic") {
            fprintf(stderr, "error: provider must be 'openai' or 'anthropic'\n");
            return 1;
        }
        RemoteKeys k = load_remote_keys();
        if (args[1] == "openai") k.openai.clear();
        else k.anthropic.clear();
        save_remote_keys(k);
        printf("Removed %s key\n", args[1].c_str());
        return 0;
    }
    fprintf(stderr, "keys: unknown subcommand '%s'\n", args[0].c_str());
    return 1;
}

static int cmd_remote(CliArgs & cli) {
    const size_t colon = cli.model.find(':');
    const std::string provider = cli.model.substr(0, colon);
    const std::string model = cli.model.substr(colon + 1);
    const std::string api_name = provider == "claude" ? "anthropic" : "openai";
    const char * env_model = std::getenv(provider == "claude" ? "ANVIL_ANTHROPIC_MODEL" : "ANVIL_OPENAI_MODEL");
    const std::string model_name = model.empty()
        ? (env_model && env_model[0] ? env_model : (api_name == "anthropic" ? "claude-sonnet-4-5" : "gpt-4o-mini"))
        : model;

    auto call = [&](const nlohmann::ordered_json & msgs, std::string & out,
                    const std::function<void(const std::string &)> & on_delta = {}) -> bool {
        return remote_chat(api_name, model_name, msgs, cli.system_prompt, cli.max_tokens, out, on_delta);
    };

    if (!cli.prompt.empty()) {
        nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
        nlohmann::ordered_json m;
        m["role"] = "user";
        m["content"] = cli.prompt;
        msgs.push_back(std::move(m));
        std::string out;
        if (!call(msgs, out, [](const std::string & d) { fputs(d.c_str(), stdout); fflush(stdout); })) {
            fprintf(stderr, "\033[31m%s\033[0m\n", out.c_str());
            return 1;
        }
        printf("\n");
        return 0;
    }

    nlohmann::ordered_json msgs = nlohmann::ordered_json::array();
    if (!cli.system_prompt.empty()) {
        nlohmann::ordered_json m;
        m["role"] = "system";
        m["content"] = cli.system_prompt;
        msgs.push_back(std::move(m));
    }
    printf("\033[1;36manvil %s session (model: %s) — Ctrl+C to exit\033[0m\n\n", api_name.c_str(), model_name.c_str());
    while (true) {
        printf("\033[32m> \033[0m");
        fflush(stdout);
        std::string line;
        if (!std::getline(std::cin, line)) break;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line.empty()) continue;
        if (line == "/exit" || line == "/quit") break;
        nlohmann::ordered_json m;
        m["role"] = "user";
        m["content"] = line;
        msgs.push_back(std::move(m));
        std::string out;
        if (!call(msgs, out, [](const std::string & d) { fputs(d.c_str(), stdout); fflush(stdout); })) {
            fprintf(stderr, "\033[31m%s\033[0m\n", out.c_str());
            if (msgs.is_array() && !msgs.empty()) msgs.erase(msgs.end() - 1);
            continue;
        }
        printf("\n\n");
        nlohmann::ordered_json a;
        a["role"] = "assistant";
        a["content"] = out;
        msgs.push_back(std::move(a));
    }
    return 0;
}

static int cmd_mcp(const std::vector<std::string> & args) {
    if (args.empty()) {
        fprintf(stderr, "usage: anvil mcp add <name> <command|url> [args...]\n");
        fprintf(stderr, "       anvil mcp list\n");
        fprintf(stderr, "       anvil mcp rm <name>\n");
        fprintf(stderr, "       anvil mcp test <name>\n");
        return 1;
    }
    const std::string & sub = args[0];
    std::vector<McpServerCfg> servers = load_mcp_servers();

    if (sub == "add" && args.size() >= 3) {
        const std::string & name = args[1];
        if (name.empty()) { fprintf(stderr, "error: server name cannot be empty\n"); return 1; }
        for (const auto & s : servers) {
            if (s.name == name) {
                fprintf(stderr, "\033[31merror: server '%s' already registered\033[0m\n", name.c_str());
                return 1;
            }
        }
        McpServerCfg cfg;
        cfg.name = name;
        cfg.command = args[2];
        for (size_t i = 3; i < args.size(); i++) cfg.args.push_back(args[i]);
        servers.push_back(cfg);
        if (!save_mcp_servers(servers)) {
            fprintf(stderr, "\033[31merror: could not save %s\033[0m\n", mcp_json_path().c_str());
            return 1;
        }
        printf("Registered MCP server '%s' (%s)\n", name.c_str(), cfg.command.c_str());
        return 0;
    }
    if (sub == "rm" && args.size() >= 2) {
        const size_t before = servers.size();
        servers.erase(std::remove_if(servers.begin(), servers.end(),
            [&](const McpServerCfg & s) { return s.name == args[1]; }), servers.end());
        if (servers.size() == before) {
            fprintf(stderr, "\033[31merror: no such server '%s'\033[0m\n", args[1].c_str());
            return 1;
        }
        if (!save_mcp_servers(servers)) {
            fprintf(stderr, "\033[31merror: could not save %s\033[0m\n", mcp_json_path().c_str());
            return 1;
        }
        printf("Removed MCP server '%s'\n", args[1].c_str());
        return 0;
    }
    if (sub == "list") {
        printf("\n\033[1;36m── MCP Servers (%zu) ──\033[0m\n", servers.size());
        for (const auto & s : servers) {
            printf("  %s %-16s %s", s.enabled ? "\033[32m[on]\033[0m" : "\033[90m[off]\033[0m",
                   s.name.c_str(), s.command.c_str());
            for (const auto & a : s.args) printf(" %s", a.c_str());
            printf("\n");
        }
        printf("\n");
        return 0;
    }
    if (sub == "test" && args.size() >= 2) {
        McpServerCfg * found = nullptr;
        for (auto & s : servers) {
            if (s.name == args[1]) { found = &s; break; }
        }
        if (!found) {
            fprintf(stderr, "\033[31merror: no such server '%s'\033[0m\n", args[1].c_str());
            return 1;
        }
        McpClient c;
        c.name = found->name;
        bool ok = false;
        if (found->command.rfind("http://", 0) == 0 || found->command.rfind("https://", 0) == 0) {
            ok = c.open_http(found->command);
        } else {
            ok = c.open_stdio(found->command, found->args);
        }
        if (ok) ok = c.start();
        if (!ok) {
            fprintf(stderr, "\033[31merror: %s: %s\033[0m\n", found->name.c_str(),
                    c.error.empty() ? "connect failed" : c.error.c_str());
            return 1;
        }
        printf("\n\033[1;36m── %s (%zu tool(s)) ──\033[0m\n", c.name.c_str(), c.tools.size());
        for (const auto & t : c.tools) {
            printf("  %-28s %s\n", t.name.c_str(), t.description.c_str());
        }
        printf("\n");
        c.stop();
        return 0;
    }
    fprintf(stderr, "mcp: unknown subcommand '%s'\n", sub.c_str());
    return 1;
}

static int cmd_serve(CliArgs & cli) {
    for (size_t i = 0; i < cli.sub_args.size(); i++) {
        const std::string & a = cli.sub_args[i];
        if ((a == "--port" || a == "-p") && i + 1 < cli.sub_args.size()) {
            parse_int(cli.sub_args[++i], cli.port);
        } else if (a == "--host" && i + 1 < cli.sub_args.size()) {
            cli.host = cli.sub_args[++i];
        } else if (a == "--model" && i + 1 < cli.sub_args.size()) {
            cli.model = cli.sub_args[++i];
        } else if (a == "--ctx" || a == "-c") {
            if (i + 1 < cli.sub_args.size()) { parse_int(cli.sub_args[++i], cli.n_ctx); }
        } else if (a == "--temp" || a == "-t") {
            if (i + 1 < cli.sub_args.size()) { float f = 0.0f; if (parse_float(cli.sub_args[++i], f)) cli.temp = f; }
        } else if (a == "--max-tokens" || a == "-n") {
            if (i + 1 < cli.sub_args.size()) { parse_int(cli.sub_args[++i], cli.max_tokens); }
        } else if (a == "--mmproj") {
            if (i + 1 < cli.sub_args.size()) cli.mmproj = cli.sub_args[++i];
        } else if (a == "--type-k") {
            if (i + 1 < cli.sub_args.size()) cli.type_k = cli.sub_args[++i];
        } else if (a == "--type-v") {
            if (i + 1 < cli.sub_args.size()) cli.type_v = cli.sub_args[++i];
        } else if (a == "--ngl" || a == "--n-gpu-layers") {
            if (i + 1 < cli.sub_args.size()) { parse_int(cli.sub_args[++i], cli.ngl); }
        } else if (a == "--flash-attn") {
            cli.flash_attn = true;
        } else if (a == "--no-flash-attn") {
            cli.flash_attn = false; cli.no_flash_attn = true;
        } else if (a == "--threads") {
            if (i + 1 < cli.sub_args.size()) { parse_int(cli.sub_args[++i], cli.n_threads); }
        } else if (a == "--api-key") {
            if (i + 1 < cli.sub_args.size()) cli.api_key = cli.sub_args[++i];
        } else if (a == "--slots") {
            if (i + 1 < cli.sub_args.size()) { parse_int(cli.sub_args[++i], cli.slots); }
        } else if (!a.empty() && a[0] != '-' && cli.model.empty()) {
            cli.model = a;
        } else {
            fprintf(stderr, "serve: unknown option '%s'\n", a.c_str());
            return 1;
        }
    }
    if (cli.model.empty()) {
        fprintf(stderr, "error: anvil serve requires --model <name>\n");
        return 1;
    }
    std::string path, friendly;
    if (!resolve_model_arg(cli.model, path, friendly)) {
        fprintf(stderr, "\033[31merror: '%s' is not a file and not a registered model\033[0m\n",
                cli.model.c_str());
        return 1;
    }
    if (!validate_gguf(path)) {
        fprintf(stderr, "\033[31merror: %s\033[0m\n", gguf_check_error(path).c_str());
        return 1;
    }
    cli.model = path;
    if (friendly.empty()) friendly = slugify(std::filesystem::path(path).stem().string());
    cli.friendly = friendly;

    const HWInfo hw = probe_hw();
    const ModelMeta meta = read_model_meta(path);
    const int max_ctx = static_cast<int>(meta.trained_ctx);

    AnvilConfig cfg = config_exists() ? load_config() : AnvilConfig{};
    if (!cfg.model.empty() && cfg.model != path) {
        std::vector<ModelEntry> models = load_models();
        if (ModelEntry * e = find_model(models, cfg.model)) {
            e->profile.apply_to(cfg);
        }
    }
    if (cli.n_ctx > 0) cfg.n_ctx = cli.n_ctx;
    if (cfg.n_ctx <= 0 && max_ctx > 0) cfg.n_ctx = max_ctx;
    if (cli.ngl >= 0) cfg.ngl = cli.ngl;
    else if (cfg.ngl < 0) cfg.ngl = derive_ngl(hw);
    if (cli.temp >= 0) cfg.temp = cli.temp;
    if (cli.n_threads > 0) cfg.n_threads = cli.n_threads;
    if (cli.flash_attn) cfg.flash_attn = true;
    if (cli.no_flash_attn) cfg.flash_attn = false;
    if (!cli.type_k.empty()) cfg.type_k = kv_type_from_name(cli.type_k);
    if (!cli.type_v.empty()) cfg.type_v = kv_type_from_name(cli.type_v);

    LlamaBackend backend;
    fprintf(stderr, "Loading model: %s ...\n", path.c_str());
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.ngl;
    LlamaModel model(llama_model_load_from_file(path.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "\033[31merror: failed to load model '%s'\033[0m\n", path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (cli.slots < 1) cli.slots = 1;
    if (cli.slots > 64) cli.slots = 64;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(cfg.n_ctx > 0 ? cfg.n_ctx : 4096);
    cparams.n_batch = cparams.n_ctx;
    cparams.n_threads = cfg.n_threads > 0 ? cfg.n_threads : hw.cpu_threads;
    cparams.flash_attn_type = cfg.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k = cfg.type_k;
    cparams.type_v = cfg.type_v;
    cparams.embeddings = true;
    cparams.n_seq_max = static_cast<uint32_t>(cli.slots) + 1;
    cparams.kv_unified = true;
    LlamaContext ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        fprintf(stderr, "\033[31merror: failed to create context\033[0m\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    llama_set_embeddings(ctx, true);

    common_chat_templates_ptr tmpls = common_chat_templates_init(model.get(), "", "", "");
    if (!tmpls) {
        fprintf(stderr, "\033[31merror: model has no usable chat template\033[0m\n");
        return 1;
    }

    ServeGenCtx g;
    g.ctx = ctx.get();
    g.model = model.get();
    g.vocab = vocab;
    g.mem = mem;
    g.tmpls = tmpls.get();
    g.cfg = cfg;
    g.model_id = friendly;
    g.slots.init(ctx.get(), mem, cli.slots);

    McpHost mcp_host;
    if (!load_mcp_servers().empty()) {
        fprintf(stderr, "  mcp        : connecting servers ...\n");
        mcp_host.connect_all();
        fprintf(stderr, "  mcp tools  : \033[32m%zu available\033[0m\n", mcp_host.tool_count());
    }

    std::string api_key = cli.api_key;
    if (api_key.empty()) {
        const char * env = std::getenv("ANVIL_API_KEY");
        if (env && env[0]) api_key = env;
    }
    g_serve_api_key = api_key;

    httplib::Server svr;
    svr.set_write_timeout(0);
    svr.set_read_timeout(0);
    svr.set_payload_max_length(64 * 1024 * 1024);

    auto cors = [](httplib::Response & res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization, x-api-key, anthropic-version");
    };
    auto auth_ok = [&](const httplib::Request & req) -> bool {
        if (g_serve_api_key.empty()) return true;
        const std::string & h = req.get_header_value("Authorization");
        if (h.rfind("Bearer ", 0) == 0 && h.substr(7) == g_serve_api_key) return true;
        if (req.get_header_value("x-api-key") == g_serve_api_key) return true;
        return false;
    };

    svr.set_pre_routing_handler([&](const httplib::Request & req, httplib::Response & res) {
        if (req.method == "OPTIONS") {
            res.status = 204;
            cors(res);
            return httplib::Server::HandlerResponse::Handled;
        }
        if (req.path == "/healthz") return httplib::Server::HandlerResponse::Unhandled;
        if (!auth_ok(req)) {
            cors(res);
            res.status = 401;
            res.set_content(R"({"error":{"message":"invalid api key","type":"authentication_error"}})", "application/json");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Get("/healthz", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(R"({"status":"ok"})", "application/json");
    });
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        nlohmann::ordered_json j;
        j["object"] = "list";
        j["data"] = nlohmann::ordered_json::array();
        nlohmann::ordered_json m;
        m["id"] = g.model_id;
        m["object"] = "model";
        m["created"] = 0;
        m["owned_by"] = "anvil";
        j["data"].push_back(m);
        for (const auto & e : load_models()) {
            nlohmann::ordered_json em;
            em["id"] = e.name;
            em["object"] = "model";
            em["created"] = 0;
            em["owned_by"] = e.source.empty() ? "anvil" : e.source;
            j["data"].push_back(em);
        }
        res.set_content(j.dump(), "application/json");
    });

    auto handle_chat = [&](const httplib::Request & req, httplib::Response & res, bool anthropic) {
        cors(res);
        nlohmann::ordered_json body;
        try {
            body = nlohmann::ordered_json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"invalid json","type":"invalid_request_error"}})", "application/json");
            return;
        }
        nlohmann::ordered_json oai = anthropic ? anthropic_to_oai(body) : body;
        if (mcp_host.tool_count() > 0) {
            nlohmann::ordered_json merged = mcp_tools_to_oai(mcp_host.all_tools());
            if (oai.contains("tools") && oai["tools"].is_array()) {
                for (const auto & t : oai["tools"]) merged.push_back(t);
            }
            oai["tools"] = std::move(merged);
        }
        const bool stream = oai.value("stream", false);
        const std::string id = serve_id(anthropic ? "msg_" : "chatcmpl-");
        ServeResult out;
        std::string err;

        if (!stream) {
            std::lock_guard<std::mutex> lock(g_serve_infer_lock);
            if (!serve_chat(g, oai, nullptr, out, err, g_serve_stop)) {
                res.status = 500;
                nlohmann::ordered_json e;
                e["error"] = {{"message", err}, {"type", "server_error"}};
                res.set_content(e.dump(), "application/json");
                return;
            }
            if (anthropic) {
                res.set_content(anthropic_message(out, id, g.model_id).dump(), "application/json");
            } else {
                nlohmann::ordered_json resp;
                resp["id"] = id;
                resp["object"] = "chat.completion";
                resp["created"] = now_unix();
                resp["model"] = g.model_id;
                nlohmann::ordered_json ch;
                ch["index"] = 0;
                ch["message"] = openai_message(out.msg);
                ch["finish_reason"] = out.finish_reason;
                resp["choices"] = nlohmann::ordered_json::array({ch});
                resp["usage"] = openai_usage(out);
                res.set_content(resp.dump(), "application/json");
            }
            return;
        }

        const bool include_usage = oai.value("stream_options", nlohmann::ordered_json::object()).value("include_usage", false);
        res.set_chunked_content_provider("text/event-stream", [&, oai, id, include_usage, out, err](size_t, httplib::DataSink & sink) mutable {
            std::lock_guard<std::mutex> lock(g_serve_infer_lock);
            auto sse = [&](const std::string & s) -> bool {
                return sink.write(s.data(), s.size());
            };
            if (anthropic) {
                std::string block_type;
                int block_idx = -1;
                int sent_tool_idx = -1;
                std::string sent_tool_id;
                std::string sent_tool_name;
                int prompt_tokens = 0;
                auto close_block = [&]() {
                    if (block_idx < 0) return;
                    nlohmann::ordered_json stop;
                    stop["type"] = "content_block_stop";
                    stop["index"] = block_idx;
                    sse(anthropic_event("content_block_stop", stop));
                    block_idx = -1;
                    block_type.clear();
                };
                auto open_block = [&](const nlohmann::ordered_json & block) {
                    block_idx++;
                    nlohmann::ordered_json start;
                    start["type"] = "content_block_start";
                    start["index"] = block_idx;
                    start["content_block"] = block;
                    sse(anthropic_event("content_block_start", start));
                };
                auto emit = [&](const common_chat_msg_diff & d) -> bool {
                    if (!d.reasoning_content_delta.empty()) {
                        if (block_type != "thinking") {
                            close_block();
                            open_block({{"type", "thinking"}, {"thinking", ""}, {"signature", ""}});
                            block_type = "thinking";
                        }
                        nlohmann::ordered_json ev;
                        ev["type"] = "content_block_delta";
                        ev["index"] = block_idx;
                        ev["delta"] = {{"type", "thinking_delta"}, {"thinking", d.reasoning_content_delta}};
                        return sse(anthropic_event("content_block_delta", ev));
                    }
                    if (!d.content_delta.empty()) {
                        if (block_type != "text") {
                            close_block();
                            open_block({{"type", "text"}, {"text", ""}});
                            block_type = "text";
                        }
                        nlohmann::ordered_json ev;
                        ev["type"] = "content_block_delta";
                        ev["index"] = block_idx;
                        ev["delta"] = {{"type", "text_delta"}, {"text", d.content_delta}};
                        return sse(anthropic_event("content_block_delta", ev));
                    }
                    if (d.tool_call_index != std::string::npos) {
                        const size_t i = d.tool_call_index;
                        if (sent_tool_idx != static_cast<int>(i)) {
                            close_block();
                            sent_tool_idx = static_cast<int>(i);
                            if (!d.tool_call_delta.id.empty()) sent_tool_id = d.tool_call_delta.id;
                            if (!d.tool_call_delta.name.empty()) sent_tool_name = d.tool_call_delta.name;
                            open_block({{"type", "tool_use"},
                                        {"id", sent_tool_id},
                                        {"name", sent_tool_name},
                                        {"input", nlohmann::ordered_json::object()}});
                            block_type = "tool_use";
                        }
                        if (!d.tool_call_delta.arguments.empty()) {
                            nlohmann::ordered_json ev;
                            ev["type"] = "content_block_delta";
                            ev["index"] = block_idx;
                            ev["delta"] = {{"type", "input_json_delta"}, {"partial_json", d.tool_call_delta.arguments}};
                            return sse(anthropic_event("content_block_delta", ev));
                        }
                    }
                    return true;
                };
                nlohmann::ordered_json start_msg;
                start_msg["type"] = "message_start";
                start_msg["message"] = {{"id", id}, {"type", "message"}, {"role", "assistant"},
                                        {"content", nlohmann::ordered_json::array()}, {"model", g.model_id},
                                        {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                                        {"usage", {{"input_tokens", 0}, {"output_tokens", 0}}}};
                sse(anthropic_event("message_start", start_msg));
                const bool ok = serve_chat(g, oai, emit, out, err, g_serve_stop, [&](int n) { prompt_tokens = n; });
                close_block();
                if (!ok) {
                    nlohmann::ordered_json ev;
                    ev["type"] = "error";
                    ev["error"] = {{"type", "server_error"}, {"message", err}};
                    sse(anthropic_event("error", ev));
                } else {
                    nlohmann::ordered_json delta_ev;
                    delta_ev["type"] = "message_delta";
                    delta_ev["delta"] = {{"stop_reason", anthropic_stop_reason(out.finish_reason)}, {"stop_sequence", nullptr}};
                    delta_ev["usage"] = {{"output_tokens", out.completion_tokens}};
                    sse(anthropic_event("message_delta", delta_ev));
                    sse(anthropic_event("message_stop", {{"type", "message_stop"}}));
                }
                sink.done();
                return true;
            }
            nlohmann::ordered_json start;
            start["id"] = id;
            start["object"] = "chat.completion.chunk";
            start["created"] = now_unix();
            start["model"] = g.model_id;
            nlohmann::ordered_json sch;
            sch["index"] = 0;
            sch["delta"] = {{"role", "assistant"}};
            sch["finish_reason"] = nullptr;
            start["choices"] = nlohmann::ordered_json::array({sch});
            sse("data: " + start.dump() + "\n\n");
            bool first_role = false;
            auto emit = [&](const common_chat_msg_diff & d) -> bool {
                const std::string chunk = openai_stream_chunk(id, g.model_id, openai_chunk_delta(d, first_role), nullptr);
                return sse(chunk);
            };
            const bool ok = serve_chat(g, oai, emit, out, err, g_serve_stop);
            if (!ok) {
                nlohmann::ordered_json e;
                e["error"] = {{"message", err}, {"type", "server_error"}};
                sse("data: " + e.dump() + "\n\n");
            } else {
                sse(openai_stream_chunk(id, g.model_id, nlohmann::ordered_json::object(), out.finish_reason.c_str()));
                if (include_usage) {
                    nlohmann::ordered_json uc;
                    uc["id"] = id;
                    uc["object"] = "chat.completion.chunk";
                    uc["created"] = now_unix();
                    uc["model"] = g.model_id;
                    uc["choices"] = nlohmann::ordered_json::array();
                    uc["usage"] = openai_usage(out);
                    sse("data: " + uc.dump() + "\n\n");
                }
                sse("data: [DONE]\n\n");
            }
            sink.done();
            return true;
        });
    };

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        handle_chat(req, res, false);
    });
    svr.Post("/v1/messages", [&](const httplib::Request & req, httplib::Response & res) {
        handle_chat(req, res, true);
    });
    svr.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        std::lock_guard<std::mutex> lock(g_serve_infer_lock);
        nlohmann::ordered_json body;
        try {
            body = nlohmann::ordered_json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"invalid json","type":"invalid_request_error"}})", "application/json");
            return;
        }
        const std::string id = serve_id("cmpl-");
        ServeResult out;
        std::string err;
        if (!serve_completions(g, body, out, err, g_serve_stop)) {
            res.status = 500;
            nlohmann::ordered_json e;
            e["error"] = {{"message", err}, {"type", "server_error"}};
            res.set_content(e.dump(), "application/json");
            return;
        }
        nlohmann::ordered_json resp;
        resp["id"] = id;
        resp["object"] = "text_completion";
        resp["created"] = now_unix();
        resp["model"] = g.model_id;
        nlohmann::ordered_json ch;
        ch["index"] = 0;
        ch["text"] = out.msg.content;
        ch["finish_reason"] = out.finish_reason;
        resp["choices"] = nlohmann::ordered_json::array({ch});
        resp["usage"] = openai_usage(out);
        res.set_content(resp.dump(), "application/json");
    });
    svr.Post("/v1/embeddings", [&](const httplib::Request & req, httplib::Response & res) {
        cors(res);
        std::lock_guard<std::mutex> lock(g_serve_infer_lock);
        nlohmann::ordered_json body;
        try {
            body = nlohmann::ordered_json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":{"message":"invalid json","type":"invalid_request_error"}})", "application/json");
            return;
        }
        nlohmann::ordered_json out;
        std::string err;
        if (!serve_embeddings(g, body, out, err, g_serve_stop)) {
            res.status = 500;
            nlohmann::ordered_json e;
            e["error"] = {{"message", err}, {"type", "server_error"}};
            res.set_content(e.dump(), "application/json");
            return;
        }
        res.set_content(out.dump(), "application/json");
    });

    fprintf(stderr, "\033[1;32manvil serve\033[0m  model=%s  %s:%d  slots=%d\n", g.model_id.c_str(), cli.host.c_str(), cli.port, cli.slots);
    fprintf(stderr, "  GET  /v1/models\n");
    fprintf(stderr, "  POST /v1/chat/completions  (OpenAI, stream + non-stream, tools)\n");
    fprintf(stderr, "  POST /v1/messages          (Anthropic, stream + non-stream, tools)\n");
    fprintf(stderr, "  POST /v1/completions\n");
    fprintf(stderr, "  POST /v1/embeddings\n");
    fprintf(stderr, "  Ctrl+C to stop\n\n");

    g_serve_stop = 0;
    install_sigint([](int) { g_serve_stop = 1; }, false);
    std::thread t([&]() { svr.listen(cli.host, cli.port); });
    while (!g_serve_stop) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    svr.stop();
    t.join();
    printf("\nserver stopped\n");
    return 0;
}
static std::string session_path() {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories(sessions_dir(), ec);
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    if (!std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&t))) {
        std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(t));
    }
    return sessions_dir() + "/" + std::string(buf) + ".md";
}

static void export_session(const std::vector<ChatMessage> & msgs, const std::string & path) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "\033[31mcould not write session to %s\033[0m\n", path.c_str());
        return;
    }
    f << "# Anvil Chat Session\n\n";
    for (const auto & m : msgs) {
        if (m.role == "system")    f << "**[System]** " << m.content << "\n\n";
        else if (m.role == "user") f << "**[You]** " << m.content << "\n\n";
        else                       f << "**[Assistant]** " << m.content << "\n\n";
    }
    fprintf(stderr, "\033[32mSession exported to %s\033[0m\n", path.c_str());
}

static std::string token_to_str(const llama_vocab * vocab, llama_token token) {
    std::string s(16, '\0');
    int n = llama_token_to_piece(vocab, token, s.data(), static_cast<int32_t>(s.size()), 0, true);
    if (n < 0) {
        s.resize(static_cast<size_t>(-n));
        n = llama_token_to_piece(vocab, token, s.data(), static_cast<int32_t>(s.size()), 0, true);
    }
    if (n < 0) n = 0;
    s.resize(static_cast<size_t>(n));
    return s;
}

static void print_ctx_bar(int used, int total) {
    if (total <= 0) return;
    const float pct = static_cast<float>(used) / static_cast<float>(total);

    int cols = 0;
    const char * cols_env = getenv("COLUMNS");
    if (cols_env) parse_int(cols_env, cols);
    const int bar_width = cols > 0 ? cols / 2 : 0;
    if (bar_width <= 0) return;

    int filled = static_cast<int>(pct * static_cast<float>(bar_width));
    if (filled > bar_width) filled = bar_width;
    if (filled < 0) filled = 0;

    const char * color;
    if (pct < 0.5f)      color = "\033[32m";
    else if (pct < 0.8f) color = "\033[33m";
    else                 color = "\033[31m";

    fprintf(stderr, "  %sctx [", color);
    for (int i = 0; i < bar_width; i++) fprintf(stderr, i < filled ? "█" : "░");
    fprintf(stderr, "] %d%% (%d/%d)\033[0m\n",
            static_cast<int>(pct * 100.0f), used, total);
}

static llama_sampler * build_sampler_chain(
        const llama_vocab * vocab, const AnvilConfig & cfg,
        bool grammar_active, const std::string & grammar_src, int seed) {
    std::vector<std::string> seq;
    auto push_tok = [&](std::string & cur) {
        while (!cur.empty() && (cur.front() == ' ' || cur.front() == '\t')) cur.erase(cur.begin());
        while (!cur.empty() && (cur.back() == ' ' || cur.back() == '\t')) cur.pop_back();
        if (!cur.empty()) seq.push_back(cur);
        cur.clear();
    };
    if (!cfg.samplers.empty()) {
        std::string cur;
        for (const char c : cfg.samplers) {
            if (c == ';' || c == ',') {
                push_tok(cur);
            } else {
                cur += c;
            }
        }
        push_tok(cur);
    } else {
        seq = { "penalties", "top_k", "top_p", "min_p", "temp" };
        if (cfg.mirostat == 1)      seq.push_back("mirostat");
        else if (cfg.mirostat == 2) seq.push_back("mirostat_v2");
        else                        seq.push_back("dist");
    }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    for (const auto & name : seq) {
        if (name == "penalties" && cfg.repeat_last_n != 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(cfg.repeat_last_n, cfg.repeat_penalty, 0.0f, 0.0f));
        } else if (name == "top_k" && cfg.top_k > 0) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(cfg.top_k));
        } else if (name == "typical" && cfg.typical > 0.0f && cfg.typical < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_typical(cfg.typical, 1));
        } else if (name == "top_p" && cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(cfg.top_p, 1));
        } else if (name == "min_p" && cfg.min_p > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p(cfg.min_p, 1));
        } else if (name == "temp" && cfg.temp > 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg.temp));
        } else if (name == "mirostat" && cfg.mirostat == 1) {
            llama_sampler_chain_add(smpl, llama_sampler_init_mirostat(
                llama_vocab_n_tokens(vocab),
                seed >= 0 ? static_cast<uint32_t>(seed) : LLAMA_DEFAULT_SEED,
                cfg.mirostat_ent, cfg.mirostat_lr, 100));
        } else if (name == "mirostat_v2" && cfg.mirostat == 2) {
            llama_sampler_chain_add(smpl, llama_sampler_init_mirostat_v2(
                seed >= 0 ? static_cast<uint32_t>(seed) : LLAMA_DEFAULT_SEED,
                cfg.mirostat_ent, cfg.mirostat_lr));
        } else if (name == "dist") {
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed >= 0 ? static_cast<uint32_t>(seed) : LLAMA_DEFAULT_SEED));
        } else if (name == "dry" || name == "xtc") {
            fprintf(stderr, "warning: sampler '%s' is not supported in this build; skipping\n", name.c_str());
        } else if (name != "grammar") {
            fprintf(stderr, "warning: unknown sampler '%s' (known: penalties, top_k, typical, top_p, min_p, temp, mirostat, mirostat_v2, dist)\n",
                    name.c_str());
        }
    }
    if (grammar_active) {
        llama_sampler * g = llama_sampler_init_grammar(vocab, grammar_src.c_str(), "root");
        if (g) llama_sampler_chain_add(smpl, g);
    }
    return smpl;
}

static std::vector<llama_token> tokenize_render(
        const llama_vocab * vocab, const std::string & text) {
    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               nullptr, 0, false, true);
    if (n == INT32_MIN) return {};
    if (n < 0) n = -n;
    if (n <= 0) return {};
    std::vector<llama_token> toks(static_cast<size_t>(n));
    const int32_t m = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                                     toks.data(), n, false, true);
    if (m < 0) return {};
    toks.resize(static_cast<size_t>(m));
    return toks;
}


int run_chat(const CliArgs & cli, AnvilConfig cfg, const HWInfo & hw,
             std::vector<ModelEntry> & models) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.ngl;
    mparams.use_mmap = !cli.no_mmap;
    mparams.use_mlock = cli.use_mlock;
    fprintf(stderr, "Loading model: %s ...\n", cli.model.c_str());
    const auto load_start = std::chrono::steady_clock::now();
    LlamaModel model(llama_model_load_from_file(cli.model.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "\033[31merror: failed to load model '%s'\033[0m\n", cli.model.c_str());
        return 1;
    }
    const double load_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - load_start).count();

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_ctx_train = llama_model_n_ctx_train(model);
    const bool has_encoder = llama_model_has_encoder(model);
    const bool has_decoder = llama_model_has_decoder(model);

    VisionSession vision;
    if (!cli.mmproj.empty()) {
        fprintf(stderr, "Loading vision projector: %s ...\n", cli.mmproj.c_str());
        if (!vision.init(cli.mmproj, model, hw, cfg.n_threads > 0 ? cfg.n_threads : hw.cpu_threads)) {
            fprintf(stderr, "\033[31merror: failed to load vision projector '%s'\033[0m\n",
                    cli.mmproj.c_str());
            return 1;
        }
        fprintf(stderr, "  vision     : \033[32mready\033[0m\n");
    }

    RagIndex session_rag;
    if (!cli.rag_dir.empty()) {
        fprintf(stderr, "Indexing RAG corpus: %s ...\n", cli.rag_dir.c_str());
        if (!session_rag.build_dir(cli.rag_dir)) {
            fprintf(stderr, "\033[33mwarning: no indexable text files in %s; RAG disabled\033[0m\n",
                    cli.rag_dir.c_str());
        } else {
            fprintf(stderr, "  rag        : \033[32m%d chunk(s) indexed\033[0m\n",
                    static_cast<int>(session_rag.size()));
        }
    }

    std::vector<std::string> shell_history = load_history_lines();

    fprintf(stderr, "\n\033[1;36mModel Info:\033[0m\n");
    fprintf(stderr, "  trained ctx : %d tokens\n", n_ctx_train);
    fprintf(stderr, "  requested   : %d tokens\n", cfg.n_ctx);
    fprintf(stderr, "  encoder     : %s\n", has_encoder ? "yes" : "no");
    fprintf(stderr, "  decoder     : %s\n", has_decoder ? "yes" : "no");
    fprintf(stderr, "  load time   : %.2fs\n", load_sec);
    if (cfg.n_ctx > n_ctx_train && n_ctx_train > 0) {
        fprintf(stderr, "\033[33m  ⚠ WARNING: requested ctx (%d) exceeds trained ctx (%d).\033[0m\n",
                cfg.n_ctx, n_ctx_train);
        fprintf(stderr, "  Quality may degrade beyond the trained context length.\n");
    }
    fprintf(stderr, "  flash attn  : %s\n",
            cfg.flash_attn ? "\033[32menabled\033[0m" : "\033[33mdisabled\033[0m (perf will suffer)");
    fprintf(stderr, "  KV cache    : K=\033[32m%s\033[0m V=\033[32m%s\033[0m\n",
            kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));
    if (cfg.mtp) fprintf(stderr, "  MTP         : \033[33menabled\033[0m\n");
    fprintf(stderr, "  sampling    : top_k=%d top_p=%.2f repeat=%.2f temp=%.2f\n",
            cfg.top_k, cfg.top_p, cfg.repeat_penalty, cfg.temp);
    fprintf(stderr, "\n");

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = static_cast<uint32_t>(cfg.n_ctx);
    cparams.n_batch   = cfg.n_batch > 0 ? static_cast<uint32_t>(cfg.n_batch) : cparams.n_ctx;
    if (cli.n_ubatch > 0) cparams.n_ubatch = static_cast<uint32_t>(cli.n_ubatch);

    cparams.n_threads = cfg.n_threads > 0 ? cfg.n_threads : hw.cpu_threads;
    cparams.flash_attn_type = cfg.flash_attn
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k = cfg.type_k;
    cparams.type_v = cfg.type_v;
    if (cli.rope_freq_base > 0.0f)  cparams.rope_freq_base  = cli.rope_freq_base;
    if (cli.rope_freq_scale > 0.0f) cparams.rope_freq_scale = cli.rope_freq_scale;
    if (cli.rope_scaling == "linear")         cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    else if (cli.rope_scaling == "yarn")     cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
    else if (cli.rope_scaling == "longrope") cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LONGROPE;
    else if (cli.rope_scaling == "none")     cparams.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;
    if (cfg.mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    LlamaContext ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        fprintf(stderr, "\033[31merror: failed to create context\033[0m\n");
        return 1;
    }

    bool grammar_active = false;
    std::string grammar_src;
    if (!cli.grammar.empty()) {
        std::ifstream gf(cli.grammar);
        if (!gf) {
            fprintf(stderr, "\033[31merror: cannot open grammar file '%s'\033[0m\n", cli.grammar.c_str());
        } else {
            grammar_src.assign(std::istreambuf_iterator<char>(gf), std::istreambuf_iterator<char>());

            LlamaSampler probe(llama_sampler_init_grammar(vocab, grammar_src.c_str(), "root"));
            grammar_active = static_cast<bool>(probe);
            if (!grammar_active) {
                fprintf(stderr, "\033[31merror: failed to parse grammar '%s'\033[0m\n", cli.grammar.c_str());
            }
        }
    }

    LlamaSampler smpl(build_sampler_chain(vocab, cfg, grammar_active, grammar_src, cfg.seed));
    if (!smpl) {
        fprintf(stderr, "\033[31merror: failed to initialize sampler chain\033[0m\n");
        return 1;
    }
    if (grammar_active) fprintf(stderr, "  grammar     : \033[32m%s\033[0m\n", cli.grammar.c_str());

    printf("\033[1;33m%s\033[0m", ANVIL_LOGO);
    if (cli.friendly.empty()) printf("  model   : %s\n", cli.model.c_str());
    else                      printf("  model   : %s  (%s)\n", cli.friendly.c_str(), cli.model.c_str());
    printf("  backend : GPU layers=%d | flash=%s | threads=%d\n",
           cfg.ngl, cfg.flash_attn ? "on" : "off", cparams.n_threads);
    printf("  ctx     : %u tokens\n", cparams.n_ctx);
    printf("  KV      : K=%s V=%s\n", kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));
    printf("  temp    : %.2f\n", cfg.temp);
    if (cfg.mtp) printf("  spec    : MTP\n");
    if (grammar_active) printf("  grammar : %s\n", cli.grammar.c_str());
    printf("  commands: /exit /clear /stats /undo /export /model /temp <f> /ctx <n>\n\n");

    std::vector<ChatMessage> history;
    std::vector<llama_token> prev_tokens;
    Utf8Buffer utf8_buf;
    std::string pending_image;
    int total_tokens_generated = 0;
    double total_gen_time = 0.0;
    const char * tmpl = llama_model_chat_template(model, nullptr);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const llama_token bos_id = llama_vocab_bos(vocab);
    std::string sys_prompt = !cli.system_prompt.empty() ? cli.system_prompt : cfg.system_prompt;

    if (!sys_prompt.empty()) {
        history.push_back({"system", sys_prompt, ""});
    }

    auto render_conversation = [&](bool add_ass, std::string & out) -> bool {
        if (!tmpl || !tmpl[0]) return false;
        std::vector<llama_chat_message> msgs;
        msgs.reserve(history.size());
        for (const auto & m : history) {
            msgs.push_back({m.role.c_str(), m.content.c_str()});
        }
        const int32_t n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                                    add_ass, nullptr, 0);
        if (n < 0) return false;
        out.resize(static_cast<size_t>(n) + 1);
        const int32_t m = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(),
                                                    add_ass, out.data(), static_cast<int32_t>(out.size()));
        if (m < 0) return false;
        out.resize(static_cast<size_t>(m));
        return true;
    };

    llama_memory_t mem = llama_get_memory(ctx);
    auto decode_tokens = [&](const std::vector<llama_token> & toks) -> bool {
        const int32_t chunk = static_cast<int32_t>(llama_n_batch(ctx));
        for (size_t i = 0; i < toks.size(); i += static_cast<size_t>(chunk)) {
            const size_t n = std::min<size_t>(static_cast<size_t>(chunk), toks.size() - i);
            const int32_t n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
            if (n_ctx_used + static_cast<int32_t>(n) > static_cast<int32_t>(llama_n_ctx(ctx))) {
                fprintf(stderr, "\n\033[33m⚠ context window full (%d/%d)\033[0m\n",
                        n_ctx_used, static_cast<int32_t>(llama_n_ctx(ctx)));
                return false;
            }
            llama_batch batch = llama_batch_get_one(
                const_cast<llama_token *>(toks.data() + i), static_cast<int32_t>(n));
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "\033[31mdecode error\033[0m\n");
                return false;
            }
        }
        return true;
    };

    MarkdownStream md;
    auto generate = [&](std::string & response, GenStats & stats) -> bool {
        llama_sampler_reset(smpl.get());
        const auto gen_start = std::chrono::steady_clock::now();
        bool decode_ok = true;
        bool stopped = false;
        while (true) {
            const int32_t n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
            if (n_ctx_used >= static_cast<int32_t>(llama_n_ctx(ctx))) {
                fprintf(stderr, "\n\033[33m⚠ context window full (%d/%d)\033[0m\n",
                        n_ctx_used, static_cast<int32_t>(llama_n_ctx(ctx)));
                break;
            }
            if (cli.max_tokens > 0 && stats.tokens_generated >= cli.max_tokens) {
                fprintf(stderr, "\n\033[33m⚠ max tokens reached (%d)\033[0m\n", cli.max_tokens);
                break;
            }
            if (g_interrupted) {
                stopped = true;
                break;
            }
            const llama_token id = llama_sampler_sample(smpl.get(), ctx, -1);
            if (!cfg.ignore_eos && llama_vocab_is_eog(vocab, id)) break;

            const std::string piece = token_to_str(vocab, id);
            const std::string printable = utf8_buf.feed(piece);
            if (!printable.empty()) {
                md.feed(printable);
                fflush(stdout);
            }
            response += piece;
            stats.tokens_generated++;
            prev_tokens.push_back(id);

            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&id), 1);
            if (llama_decode(ctx, batch) != 0) {
                fprintf(stderr, "\033[31mdecode error\033[0m\n");
                decode_ok = false;
                break;
            }
        }
        const std::string tail = utf8_buf.flush();
        if (!tail.empty()) {
            md.feed(tail);
            fflush(stdout);
        }
        md.flush();
        stats.elapsed_sec = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - gen_start).count();
        if (stopped) {
            fprintf(stderr, "\033[33m[stopped by Ctrl+C — session preserved]\033[0m\n");
            g_interrupted = 0;
        }
        return decode_ok;
    };

    auto start_turn = [&](std::string & formatted) -> bool {
        std::vector<llama_token> all = tokenize_render(vocab, formatted);
        if (all.empty()) {
            fprintf(stderr, "\033[31mtokenization failed\033[0m\n");
            return false;
        }
        if (!prev_tokens.empty()) {

            if (prev_tokens.size() <= all.size() &&
                std::equal(prev_tokens.begin(), prev_tokens.end(), all.begin())) {
                std::vector<llama_token> delta(all.begin() + static_cast<long>(prev_tokens.size()), all.end());
                if (delta.empty()) return true;
                if (!decode_tokens(delta)) return false;
                prev_tokens = std::move(all);
                return true;
            }

            llama_memory_clear(mem, true);
            prev_tokens.clear();
        }

        std::vector<llama_token> full;
        full.reserve(all.size() + 1);
        if (add_bos && (all.empty() || all[0] != bos_id)) full.push_back(bos_id);
        full.insert(full.end(), all.begin(), all.end());
        if (!decode_tokens(full)) return false;
        prev_tokens = std::move(all);
        return true;
    };

    auto finish_turn = [&](const std::string & resp) {
        if (!resp.empty()) {
            history.push_back({"assistant", resp, ""});
        }
    };

    auto reload_context = [&](uint32_t new_ctx) -> bool {

        std::string formatted;
        const bool have_render = render_conversation(false, formatted);
        std::string text;
        if (have_render && !formatted.empty()) {
            text = formatted;
        } else if (!history.empty()) {

            for (const auto & m : history) text += m.content + "\n";
        }

        std::vector<llama_token> all = tokenize_render(vocab, text);
        if (!all.empty() && all.size() + (add_bos ? 1u : 0u) > new_ctx) {
            fprintf(stderr, "\033[31merror: conversation (%zu tokens) does not fit in %u\033[0m\n",
                    all.size(), new_ctx);
            return false;
        }

        cparams.n_ctx    = new_ctx;
        cparams.n_batch  = new_ctx;
        LlamaContext new_ctx_obj(llama_init_from_model(model, cparams));
        if (!new_ctx_obj) {
            fprintf(stderr, "\033[31merror: failed to recreate context at %u tokens\033[0m\n", new_ctx);
            return false;
        }
        ctx = std::move(new_ctx_obj);
        mem = llama_get_memory(ctx);
        prev_tokens.clear();
        if (all.empty()) return true;

        std::vector<llama_token> full;
        full.reserve(all.size() + 1);
        if (add_bos && (all.empty() || all[0] != bos_id)) full.push_back(bos_id);
        full.insert(full.end(), all.begin(), all.end());
        if (!decode_tokens(full)) {
            fprintf(stderr, "\033[31merror: failed to re-decode conversation\033[0m\n");
            return false;
        }
        prev_tokens = std::move(all);
        return true;
    };

    auto persist_chat_setting = [&](const std::string & key, const nlohmann::json & value) {
        if (cli.friendly.empty()) return;
        if (ModelEntry * e = find_model(models, cli.friendly)) {
            e->profile.set(key, value);
            if (save_models(models)) {
                printf("  └─ saved to profile '%s' (%s)\n", cli.friendly.c_str(), key.c_str());
            } else {
                printf("  └─ \033[33mwarning: could not save to profile '%s'\033[0m\n",
                       cli.friendly.c_str());
            }
        }
    };

    std::string session_path_str;
    if (cli.resume) {
        session_path_str = latest_session_for(cli.friendly.empty() ? slugify(std::filesystem::path(cli.model).stem().string()) : cli.friendly);
        if (!session_path_str.empty()) {
            std::vector<ChatMessage> loaded;
            if (load_session(session_path_str, loaded)) {
                history = std::move(loaded);
                fprintf(stderr, "\033[1;36mResumed session: %s (%zu message(s))\033[0m\n",
                        session_path_str.c_str(), history.size());
                llama_memory_clear(mem, true);
                prev_tokens.clear();
                bool has_tool_msgs = false;
                for (const auto & m : history) {
                    if (m.role == "tool" || !m.tool_calls_json.empty()) { has_tool_msgs = true; break; }
                }
                if (!has_tool_msgs && !history.empty()) {
                    std::string formatted;
                    if (render_conversation(false, formatted) && !formatted.empty()) {
                        std::vector<llama_token> all = tokenize_render(vocab, formatted);
                        if (!all.empty()) {
                            std::vector<llama_token> full;
                            full.reserve(all.size() + 1);
                            if (add_bos && (all.empty() || all[0] != bos_id)) full.push_back(bos_id);
                            full.insert(full.end(), all.begin(), all.end());
                            if (decode_tokens(full)) prev_tokens = std::move(all);
                        }
                    }
                }
            } else {
                session_path_str.clear();
            }
        }
    }
    if (session_path_str.empty()) {
        session_path_str = new_session_path(
            cli.friendly.empty() ? slugify(std::filesystem::path(cli.model).stem().string()) : cli.friendly);
    }

    McpHost mcp_host;
    common_chat_templates_ptr mcp_tmpls;
    ServeGenCtx mcp_g;
    bool mcp_active = false;
    if (!load_mcp_servers().empty()) {
        mcp_tmpls = common_chat_templates_init(model.get(), "", "", "");
        if (mcp_tmpls) {
            mcp_g.ctx = ctx.get();
            mcp_g.model = model.get();
            mcp_g.vocab = vocab;
            mcp_g.mem = mem;
            mcp_g.tmpls = mcp_tmpls.get();
            mcp_g.cfg = cfg;
            mcp_g.model_id = cli.friendly;
            fprintf(stderr, "  mcp        : connecting servers ...\n");
            mcp_host.connect_all();
            mcp_active = mcp_host.tool_count() > 0;
            if (mcp_active) {
                mcp_g.slots.init(ctx.get(), mem, 1);
                fprintf(stderr, "  mcp tools  : \033[32m%zu available\033[0m\n", mcp_host.tool_count());
            } else {
                fprintf(stderr, "  mcp tools  : \033[33mnone available\033[0m\n");
            }
        }
    }

    auto run_mcp_turn = [&](const std::string & user_msg) -> bool {
        history.push_back({"user", user_msg, ""});
        constexpr int MAX_ROUNDS = 8;
        for (int round = 0; round < MAX_ROUNDS; round++) {
            nlohmann::ordered_json body;
            body["messages"] = chat_history_to_oai(history);
            body["tools"] = mcp_tools_to_oai(mcp_host.all_tools());
            body["stream"] = false;
            mcp_g.slots.reset_all();
            prev_tokens.clear();
            ServeResult out;
            std::string err;
            auto emit = [&](const common_chat_msg_diff & d) -> bool {
                if (!d.content_delta.empty()) md.feed(d.content_delta);
                return true;
            };
            const bool ok = serve_chat(mcp_g, body, emit, out, err, g_interrupted);
            md.flush();
            if (!ok) {
                fprintf(stderr, "\033[31m[mcp] %s\033[0m\n", err.c_str());
                history.pop_back();
                return false;
            }
            if (!out.msg.tool_calls.empty()) {
                ChatMessage am;
                am.role = "assistant";
                am.content = out.msg.content;
                am.tool_calls_json = chat_tool_calls_to_json(out.msg.tool_calls);
                history.push_back(std::move(am));
                for (const auto & tc : out.msg.tool_calls) {
                    nlohmann::json args = nlohmann::json::object();
                    if (!tc.arguments.empty()) {
                        try { args = nlohmann::json::parse(tc.arguments); }
                        catch (const nlohmann::json::exception &) {}
                    }
                    printf("\n\033[1;36m  \xe2\x9a\x92 %s\033[0m %s\n", tc.name.c_str(), tc.arguments.c_str());
                    fflush(stdout);
                    std::string result;
                    mcp_host.call(tc.name, args, result);
                    const std::string shown = result.size() > 400
                        ? result.substr(0, 400) + " \xe2\x80\xa6" : result;
                    printf("  \xe2\x94\x94\xe2\x94\x80 %s\n\n", shown.c_str());
                    fflush(stdout);
                    ChatMessage tr;
                    tr.role = "tool";
                    tr.content = result;
                    tr.tool_call_id = tc.id;
                    history.push_back(std::move(tr));
                }
                save_session(session_path_str, history);
                continue;
            }
            finish_turn(out.msg.content);
            save_session(session_path_str, history);
            return true;
        }
        fprintf(stderr, "\033[33m[mcp] tool loop exceeded %d rounds; stopping\033[0m\n", MAX_ROUNDS);
        return true;
    };

    auto summarize_quiet = [&](const std::string & prompt, std::string & out) -> bool {
        std::vector<llama_token> toks = tokenize_render(vocab, prompt);
        if (toks.empty()) return false;
        llama_memory_clear(mem, true);
        prev_tokens.clear();
        if (!decode_tokens(toks)) return false;
        prev_tokens = std::move(toks);
        AnvilConfig sum_cfg = cfg;
        sum_cfg.temp = 0.3f;
        LlamaSampler sum_smpl(build_sampler_chain(vocab, sum_cfg, false, "", 42));
        if (!sum_smpl) return false;
        llama_sampler_reset(sum_smpl.get());
        std::string acc;
        int n = 0;
        while (n < 800) {
            const int32_t used = llama_memory_seq_pos_max(mem, 0) + 1;
            if (used >= static_cast<int32_t>(llama_n_ctx(ctx))) break;
            const llama_token id = llama_sampler_sample(sum_smpl.get(), ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;
            acc += token_to_str(vocab, id);
            n++;
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&id), 1);
            if (llama_decode(ctx, batch) != 0) break;
        }
        out = acc;
        return !out.empty();
    };

    auto compact_history = [&]() -> bool {
        if (history.size() <= 6) return false;
        std::string sys;
        size_t first_keep = 0;
        for (size_t i = 0; i < history.size(); i++) {
            if (history[i].role == "system") {
                sys = history[i].content;
                first_keep = i + 1;
            }
        }
        if (first_keep >= history.size()) return false;
        const size_t keep_from = history.size() > 6 ? history.size() - 5 : first_keep + 1;
        if (keep_from <= first_keep) return false;
        std::string dump;
        for (size_t i = first_keep; i < keep_from; i++) {
            const auto & m = history[i];
            const char * tag = m.role == "assistant" ? "Assistant"
                             : m.role == "tool" ? "Tool result" : "User";
            dump += std::string(tag) + ": " + m.content + "\n";
        }
        if (dump.size() < 64) return false;
        const std::string prompt =
            "Summarize this conversation into one concise paragraph. Preserve every concrete fact, number, decision, and open task. Omit greetings and small talk.\n\n" +
            dump + "\nSummary:";
        std::string summary;
        if (!summarize_quiet(prompt, summary)) return false;
        if (summary.size() > 2400) summary.resize(2400);
        std::vector<ChatMessage> kept;
        if (!sys.empty()) {
            kept.push_back({"system", sys, ""});
        }
        kept.push_back({"system", "[Summary of earlier conversation]\n" + summary, ""});
        for (size_t i = keep_from; i < history.size(); i++) kept.push_back(history[i]);
        history = std::move(kept);
        llama_memory_clear(mem, true);
        prev_tokens.clear();
        fprintf(stderr, "\n\033[1;36m── Context compacted ──\033[0m\n");
        fprintf(stderr, "  summary : %zu chars\n  kept    : %zu message(s)\n\n",
                summary.size(), history.size());
        save_session(session_path_str, history);
        return true;
    };

    if (cli.prompt.empty() || cli.interactive) {
        std::string first_prompt = cli.prompt;

        while (true) {
            g_interrupted = 0;
            int32_t n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
            const uint32_t n_ctx_total = llama_n_ctx(ctx);
            if (n_ctx_used > 0 && n_ctx_total > 0 &&
                static_cast<double>(n_ctx_used) / n_ctx_total > 0.75 &&
                history.size() > 6) {
                if (!compact_history()) {
                    fprintf(stderr, "\033[33m⚠ context at %d/%u; compaction unavailable\033[0m\n",
                            n_ctx_used, static_cast<int>(n_ctx_total));
                }
                n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
            }
            if (n_ctx_used > 0) print_ctx_bar(n_ctx_used, static_cast<int>(llama_n_ctx(ctx)));
            printf("\033[32m> \033[0m");
            fflush(stdout);

            std::string user_input;
            if (!first_prompt.empty()) {
                user_input = first_prompt;
                first_prompt.clear();
            } else if (!std::getline(std::cin, user_input)) break;
            while (!user_input.empty() && (user_input.back() == '\n' || user_input.back() == '\r'))
                user_input.pop_back();
            if (g_interrupted) break;
            if (user_input.empty()) continue;

            if (user_input == "/exit" || user_input == "/quit") break;

            if (user_input == "/clear") {
                history.clear();
                llama_memory_clear(mem, true);
                prev_tokens.clear();
                if (!sys_prompt.empty()) {
                    history.push_back({"system", sys_prompt, ""});
                }
                total_tokens_generated = 0;
                total_gen_time = 0.0;
                save_session(session_path_str, history);
                printf("Chat cleared.\n\n");
                continue;
            }

            if (user_input == "/new") {
                history.clear();
                llama_memory_clear(mem, true);
                prev_tokens.clear();
                total_tokens_generated = 0;
                total_gen_time = 0.0;
                if (!sys_prompt.empty()) {
                    history.push_back({"system", sys_prompt, ""});
                }
                session_path_str = new_session_path(
                    cli.friendly.empty() ? slugify(std::filesystem::path(cli.model).stem().string()) : cli.friendly);
                printf("New session started.\n\n");
                continue;
            }

            if (user_input == "/regenerate" || user_input.rfind("/regenerate ", 0) == 0) {
                if (history.size() >= 2 &&
                    history.back().role == "assistant" &&
                    history[history.size() - 2].role == "user") {
                    history.pop_back();
                    llama_memory_clear(mem, true);
                    prev_tokens.clear();
                    std::string formatted;
                    if (render_conversation(true, formatted) && start_turn(formatted)) {
                        std::string resp;
                        GenStats stats;
                        generate(resp, stats);
                        printf("\n");
                        if (stats.tokens_generated > 0) {
                            fprintf(stderr, "  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                                    stats.tokens_generated, stats.tps(), stats.elapsed_sec);
                        }
                        total_tokens_generated += stats.tokens_generated;
                        total_gen_time += stats.elapsed_sec;
                        finish_turn(resp);
                        save_session(session_path_str, history);
                        printf("\n");
                    }
                } else {
                    printf("Nothing to regenerate.\n\n");
                }
                continue;
            }

            if (user_input.rfind("/edit ", 0) == 0) {
                if (history.size() >= 2 &&
                    history.back().role == "assistant" &&
                    history[history.size() - 2].role == "user") {
                    const std::string new_text = user_input.substr(6);
                    history[history.size() - 2].content = new_text;
                    history.pop_back();
                    llama_memory_clear(mem, true);
                    prev_tokens.clear();
                    std::string formatted;
                    if (render_conversation(true, formatted) && start_turn(formatted)) {
                        std::string resp;
                        GenStats stats;
                        generate(resp, stats);
                        printf("\n");
                        if (stats.tokens_generated > 0) {
                            fprintf(stderr, "  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                                    stats.tokens_generated, stats.tps(), stats.elapsed_sec);
                        }
                        total_tokens_generated += stats.tokens_generated;
                        total_gen_time += stats.elapsed_sec;
                        finish_turn(resp);
                        save_session(session_path_str, history);
                        printf("\n");
                    }
                } else {
                    printf("Nothing to edit.\n\n");
                }
                continue;
            }

            if (user_input.rfind("/image ", 0) == 0) {
                if (vision.ctx == nullptr) {
                    printf("Vision not loaded. Run with --mmproj <file>.\n\n");
                    continue;
                }
                std::string img = expand_home(user_input.substr(7));
                if (!std::filesystem::exists(img)) {
                    printf("No such image: %s\n\n", img.c_str());
                    continue;
                }
                pending_image = img;
                printf("Image attached: %s — send a message to use it (or /image again to replace).\n\n",
                       img.c_str());
                continue;
            }

            if (user_input == "/image") {
                printf("Usage: /image <path>\n\n");
                continue;
            }

            if (user_input == "/ingest clear") {
                session_rag.chunks.clear();
                session_rag.idf.clear();
                session_rag.postings.clear();
                session_rag.sources.clear();
                printf("Session index cleared.\n\n");
                continue;
            }
            if (user_input == "/ingest") {
                printf("Session index: %zu chunk(s) from %zu source(s)\n\n",
                       session_rag.size(), session_rag.sources.size());
                for (const auto & s : session_rag.sources) printf("  - %s\n", s.c_str());
                printf("\nUsage: /ingest <file|dir|url>\n\n");
                continue;
            }
            if (user_input.rfind("/ingest ", 0) == 0) {
                const std::string target = expand_home(user_input.substr(8));
                bool ok = false;
                if (target.rfind("http://", 0) == 0 || target.rfind("https://", 0) == 0) {
                    printf("Fetching %s ...\n", target.c_str());
                    fflush(stdout);
                    ok = session_rag.add_url(target);
                } else if (std::filesystem::is_directory(target)) {
                    ok = session_rag.build_dir(target);
                } else if (std::filesystem::is_regular_file(target)) {
                    ok = session_rag.add_file(target);
                } else {
                    printf("No such file or directory: %s\n\n", target.c_str());
                    continue;
                }
                if (ok) {
                    printf("Indexed %s — %zu chunk(s), %zu source(s).\n\n",
                           target.c_str(), session_rag.size(), session_rag.sources.size());
                } else {
                    printf("Nothing indexable found in %s\n\n", target.c_str());
                }
                continue;
            }

            if (user_input.rfind("/system ", 0) == 0 || user_input == "/system") {
                if (user_input == "/system") {
                    const auto names = list_presets();
                    if (names.empty()) {
                        printf("No presets in %s.\n", presets_dir().c_str());
                    } else {
                        printf("Presets in %s:\n", presets_dir().c_str());
                        for (const auto & n : names) printf("  @%s\n", n.c_str());
                        printf("Use: /system @<name> or /system <text>\n");
                    }
                    printf("\n");
                    continue;
                }
                const std::string arg = user_input.substr(8);
                const std::string sp = resolve_system_prompt(arg);
                if (!sp.empty() || (arg.size() > 1 && arg[0] == '@')) {
                    if (!sp.empty()) {
                        history.erase(std::remove_if(history.begin(), history.end(),
                            [](const ChatMessage & m) { return m.role == "system"; }),
                            history.end());
                        history.insert(history.begin(), ChatMessage{"system", sp, ""});
                        sys_prompt = sp;
                        llama_memory_clear(mem, true);
                        prev_tokens.clear();
                        printf("System prompt set. Next message re-decodes the conversation.\n\n");
                        continue;
                    }
                    printf("Preset not found.\n\n");
                    continue;
                }
                printf("System prompt set to raw text.\n\n");
                continue;
            }

            if (user_input == "/history") {
                if (shell_history.empty()) {
                    printf("No shell history yet.\n\n");
                } else {
                    printf("\n\033[1;36m── Shell History (%zu) ──\033[0m\n", shell_history.size());
                    for (size_t i = 0; i < shell_history.size(); i++) {
                        printf("  %4zu  %s\n", i + 1, shell_history[i].c_str());
                    }
                    printf("\n");
                }
                continue;
            }

            if (user_input.rfind("/search ", 0) == 0) {
                const std::string term = user_input.substr(8);
                printf("\n\033[1;36m── History matching '%s' ──\033[0m\n", term.c_str());
                int hits = 0;
                for (size_t i = 0; i < shell_history.size(); i++) {
                    if (shell_history[i].find(term) != std::string::npos) {
                        printf("  %4zu  %s\n", i + 1, shell_history[i].c_str());
                        hits++;
                    }
                }
                if (hits == 0) printf("  (no matches)\n");
                printf("\n");
                continue;
            }

            if (user_input == "/stats") {
                printf("\n\033[1;36m── Session Stats ──\033[0m\n");
                printf("  turns           : %zu\n", history.size());
                printf("  tokens generated: %d\n", total_tokens_generated);
                printf("  total gen time  : %.2fs\n", total_gen_time);
                if (total_gen_time > 0)
                    printf("  avg speed       : %.1f t/s\n", total_tokens_generated / total_gen_time);
                const int32_t n_used = llama_memory_seq_pos_max(mem, 0) + 1;
                printf("  ctx used        : %d / %u (%.1f%%)\n", n_used,
                       cparams.n_ctx, 100.0 * n_used / cparams.n_ctx);
                printf("  KV cache        : K=%s V=%s\n", kv_type_short(cfg.type_k), kv_type_short(cfg.type_v));
                printf("  temp            : %.2f\n", cfg.temp);
                printf("\n");
                continue;
            }

            if (user_input == "/undo") {
                if (history.size() >= 2 &&
                    history.back().role == "assistant" &&
                    history[history.size() - 2].role == "user") {
                    history.pop_back();
                    history.pop_back();
                    llama_memory_clear(mem, true);
                    prev_tokens.clear();
                    printf("Undid last turn. Next message will re-decode the conversation.\n\n");
                } else {
                    printf("Nothing to undo.\n\n");
                }
                continue;
            }

            if (user_input == "/export") {
                const std::string path = session_path();
                export_session(history, path);
                continue;
            }

            if (user_input == "/model") {
                printf("\n\033[1;36m── Model Info ──\033[0m\n");
                if (!cli.friendly.empty()) printf("  name       : %s\n", cli.friendly.c_str());
                printf("  file       : %s\n", cli.model.c_str());
                const int32_t alen = llama_model_desc(model, nullptr, 0);
                if (alen > 0) {
                    std::string arch_buf(static_cast<size_t>(alen) + 1, '\0');
                    llama_model_desc(model, arch_buf.data(), arch_buf.size());
                    printf("  arch       : %s\n", arch_buf.c_str());
                } else {
                    printf("  arch       : unknown\n");
                }
                printf("  trained ctx: %d\n", n_ctx_train);
                printf("  encoder    : %s\n", has_encoder ? "yes" : "no");
                printf("  decoder    : %s\n", has_decoder ? "yes" : "no");
                printf("  ngl        : %d\n", cfg.ngl);
                printf("\n");
                continue;
            }

            if (user_input.rfind("/temp ", 0) == 0) {
                float new_temp = 0.0f;
                if (!parse_float(user_input.substr(6), new_temp) || new_temp < 0.0f || new_temp > MAX_TEMP) {
                    printf("Invalid temperature.\n\n");
                    continue;
                }
                cfg.temp = new_temp;
                smpl.reset(build_sampler_chain(vocab, cfg, grammar_active, grammar_src));
                printf("Temperature set to %.2f\n", new_temp);

                persist_chat_setting("temp", std::round(static_cast<double>(new_temp) * 100.0) / 100.0);
                printf("\n");
                continue;
            }

            if (user_input == "/ctx") {
                const int32_t n_used = llama_memory_seq_pos_max(mem, 0) + 1;
                print_ctx_bar(n_used, static_cast<int>(llama_n_ctx(ctx)));
                printf("\n");
                continue;
            }

            if (user_input.rfind("/ctx ", 0) == 0) {
                int new_ctx = 0;
                if (!parse_int(user_input.substr(5), new_ctx) || new_ctx < 1 || new_ctx > MAX_CTX) {
                    printf("Invalid context size (positive integer, no artificial cap).\n\n");
                    continue;
                }
                if (static_cast<uint32_t>(new_ctx) == cparams.n_ctx) {
                    printf("Context is already %d tokens.\n\n", new_ctx);
                    continue;
                }
                if (n_ctx_train > 0 && new_ctx > n_ctx_train) {
                    printf("\033[33m⚠ %d exceeds trained context (%d); quality may degrade.\033[0m\n",
                           new_ctx, n_ctx_train);
                }
                if (!reload_context(static_cast<uint32_t>(new_ctx))) {
                    printf("Context left unchanged.\n\n");
                    continue;
                }
                printf("Context resized to %d tokens; conversation reloaded.\n", new_ctx);
                persist_chat_setting("n_ctx", new_ctx);
                printf("\n");
                continue;
            }

            if (user_input[0] == '/') {
                printf("Unknown command: %s\n", user_input.c_str());
                printf("Available: /exit /clear /stats /undo /export /model /temp <f> /ctx <n>\n\n");
                continue;
            }

            if (user_input.rfind("gpt: ", 0) == 0 || user_input.rfind("claude: ", 0) == 0) {
                const size_t sp = user_input.find(' ');
                const std::string provider = user_input.substr(0, sp);
                const std::string text = user_input.substr(sp + 1);
                const std::string api_name = provider == "claude" ? "anthropic" : "openai";
                const char * env_model = std::getenv(provider == "claude" ? "ANVIL_ANTHROPIC_MODEL" : "ANVIL_OPENAI_MODEL");
                const std::string model_name = env_model && env_model[0]
                    ? env_model : (api_name == "anthropic" ? "claude-sonnet-4-5" : "gpt-4o-mini");
                history.push_back({"user", text, ""});
                std::string resp;
                if (remote_chat(api_name, model_name, chat_history_to_oai(history), sys_prompt,
                                cli.max_tokens, resp,
                                [&](const std::string & d) { md.feed(d); })) {
                    md.flush();
                    finish_turn(resp);
                } else {
                    md.flush();
                    fprintf(stderr, "\033[31m%s\033[0m\n", resp.c_str());
                    history.pop_back();
                }
                save_session(session_path_str, history);
                append_history_line(user_input);
                continue;
            }

            std::string user_msg = user_input;
            std::string rag_ctx;
            if (!session_rag.empty()) {
                rag_ctx = session_rag.context_for(user_input, 3);
                if (!rag_ctx.empty()) {
                    user_msg = "Relevant context:\n\n" + rag_ctx + "Question: " + user_input;
                }
            }
            if (mcp_active && pending_image.empty()) {
                if (!run_mcp_turn(user_msg)) break;
                append_history_line(user_input);
                continue;
            }
            if (!pending_image.empty() && vision.ctx != nullptr) {
                history.push_back({"user", user_msg, pending_image});
                pending_image.clear();
                std::string formatted;
                if (!render_conversation(true, formatted)) {
                    history.pop_back();
                    llama_memory_clear(mem, true);
                    prev_tokens.clear();
                    std::string raw = user_msg + "\n";
                    std::vector<llama_token> raw_toks;
                    raw_toks.reserve(raw.size() + 1);
                    if (add_bos) raw_toks.push_back(bos_id);
                    const auto extra = tokenize_render(vocab, raw);
                    raw_toks.insert(raw_toks.end(), extra.begin(), extra.end());
                    if (!decode_tokens(raw_toks)) continue;
                    prev_tokens = extra;
                    std::string resp;
                    GenStats stats;
                    generate(resp, stats);
                    printf("\n");
                    finish_turn(resp);
                    save_session(session_path_str, history);
                    continue;
                }
                llama_memory_clear(mem, true);
                prev_tokens.clear();
                llama_pos new_past = 0;
                if (!vision.eval_image_turn(ctx.get(), formatted, history.back().image_path,
                                            &new_past, static_cast<int32_t>(llama_n_batch(ctx)))) {
                    fprintf(stderr, "\033[31merror: vision encode failed\033[0m\n");
                    history.pop_back();
                    continue;
                }
                std::string resp;
                GenStats stats;
                generate(resp, stats);
                printf("\n");
                if (stats.tokens_generated > 0) {
                    fprintf(stderr, "  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                            stats.tokens_generated, stats.tps(), stats.elapsed_sec);
                }
                total_tokens_generated += stats.tokens_generated;
                total_gen_time += stats.elapsed_sec;
                finish_turn(resp);
                save_session(session_path_str, history);
                continue;
            }
            history.push_back({"user", user_msg, ""});
            std::string formatted;
            if (!render_conversation(true, formatted)) {

                history.pop_back();
                llama_memory_clear(mem, true);
                prev_tokens.clear();
                std::string raw = user_input + "\n";
                std::vector<llama_token> raw_toks;
                raw_toks.reserve(raw.size() + 1);
                if (add_bos) raw_toks.push_back(bos_id);
                const auto extra = tokenize_render(vocab, raw);
                raw_toks.insert(raw_toks.end(), extra.begin(), extra.end());
                if (!decode_tokens(raw_toks)) continue;
                prev_tokens = extra;
                std::string resp;
                GenStats stats;
                generate(resp, stats);
                printf("\n");
                finish_turn(resp);
                save_session(session_path_str, history);
                append_history_line(user_input);
                continue;
            }
            if (!start_turn(formatted)) continue;

            std::string resp;
            GenStats stats;
            generate(resp, stats);
            printf("\n");
            if (stats.tokens_generated > 0) {
                fprintf(stderr, "  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                        stats.tokens_generated, stats.tps(), stats.elapsed_sec);
            }
            total_tokens_generated += stats.tokens_generated;
            total_gen_time += stats.elapsed_sec;
            finish_turn(resp);
            save_session(session_path_str, history);
            append_history_line(user_input);
        }
    } else {
        if (mcp_active && cli.image.empty()) {
            run_mcp_turn(cli.prompt);
            printf("\nExiting.\n");
            return 0;
        }

        std::string shot_prompt = cli.prompt;
        RagIndex shot_rag;
        if (!cli.rag_dir.empty() && shot_rag.build_dir(cli.rag_dir)) {
            const std::string ctx = shot_rag.context_for(cli.prompt, 3);
            if (!ctx.empty()) shot_prompt = "Relevant context:\n\n" + ctx + "Question: " + cli.prompt;
        }
        history.push_back({"user", shot_prompt, cli.image});
        std::string formatted;
        if (!cli.image.empty() && vision.ctx != nullptr) {
            if (!std::filesystem::exists(cli.image)) {
                fprintf(stderr, "\033[31merror: no such image: %s\033[0m\n", cli.image.c_str());
                return 1;
            }
            llama_memory_clear(mem, true);
            llama_pos new_past = 0;
            if (!render_conversation(true, formatted)) {
                fprintf(stderr, "\033[31merror: template rendering failed\033[0m\n");
                return 1;
            }
            if (!vision.eval_image_turn(ctx.get(), formatted, cli.image,
                                        &new_past, static_cast<int32_t>(llama_n_batch(ctx)))) {
                fprintf(stderr, "\033[31merror: vision encode failed\033[0m\n");
                return 1;
            }
            std::string resp;
            GenStats stats;
            generate(resp, stats);
            printf("\n");
            if (stats.tokens_generated > 0) {
                fprintf(stderr, "\n  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                        stats.tokens_generated, stats.tps(), stats.elapsed_sec);
            }
            finish_turn(resp);
            save_session(session_path_str, history);
            printf("\nExiting.\n");
            return 0;
        }
        bool have_render = render_conversation(true, formatted);
        std::string prompt_text = have_render ? formatted : (cli.prompt + "\n");
        if (have_render && formatted.empty()) prompt_text = cli.prompt + "\n";

        std::vector<llama_token> all = tokenize_render(vocab, prompt_text);
        if (all.empty()) {
            fprintf(stderr, "\033[31mtokenization failed\033[0m\n");
            return 1;
        }
        std::vector<llama_token> full;
        full.reserve(all.size() + 1);
        if (add_bos && (all.empty() || all[0] != bos_id)) full.push_back(bos_id);
        full.insert(full.end(), all.begin(), all.end());
        if (!decode_tokens(full)) return 1;
        prev_tokens = std::move(all);

        std::string resp;
        GenStats stats;
        generate(resp, stats);
        printf("\n");
        if (stats.tokens_generated > 0) {
            fprintf(stderr, "\n  \033[2m[%d tokens, %.1f t/s, %.1fs]\033[0m\n",
                    stats.tokens_generated, stats.tps(), stats.elapsed_sec);
        }
        finish_turn(resp);
        save_session(session_path_str, history);
        append_history_line(cli.prompt);
    }
    printf("\nExiting.\n");
    return 0;
}


struct EvalSample {
    std::string question;
    std::vector<std::string> choices;
    std::string answer;
};

static std::vector<EvalSample> eval_fetch_rows(const std::string & dataset,
                                                 const std::string & config,
                                                 const std::string & split, int limit,
                                                 const std::string & subject = "") {
    std::vector<EvalSample> out;
    const std::string base = "https://datasets-server.huggingface.co/rows?dataset=" + dataset +
                             "&config=" + config + "&split=" + split;
    size_t offset = 0;
    while (static_cast<int>(out.size()) < limit) {
        const std::string body = http_get(base + "&offset=" + std::to_string(offset) + "&length=100");
        if (body.empty()) break;
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(body);
        } catch (const nlohmann::json::exception &) { break; }
        if (!j.contains("rows") || j["rows"].empty()) break;
        for (const auto & r : j["rows"]) {
            const auto & row = r["row"];
            if (!subject.empty() && row.value("subject", std::string()) != subject) continue;
            EvalSample s;
            s.question = row.value("question", std::string());
            if (row.contains("choices")) {
                for (const auto & c : row["choices"]) {
                    s.choices.push_back(c.get<std::string>());
                }
                if (row.contains("answer") && row["answer"].is_number_integer()) {
                    const int idx = row["answer"].get<int>();
                    if (idx >= 0 && idx < 4) {
                        s.answer = std::string(1, static_cast<char>('A' + idx));
                    }
                }
            } else {
                s.answer = row.value("answer", std::string());
            }
            if (s.question.empty()) continue;
            out.push_back(std::move(s));
            if (static_cast<int>(out.size()) >= limit) break;
        }
        if (static_cast<int>(out.size()) >= limit) break;
        offset += 100;
    }
    return out;
}

static std::vector<std::string> eval_csv_split(const std::string & line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_q = false;
    for (size_t i = 0; i < line.size(); i++) {
        const char c = line[i];
        if (c == '"') {
            if (in_q && i + 1 < line.size() && line[i + 1] == '"') {
                cur += '"';
                i++;
            } else {
                in_q = !in_q;
            }
        } else if (c == ',' && !in_q) {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

static std::string eval_normalize(const std::string & s) {
    std::string out;
    for (const unsigned char c : s) {
        if (std::isdigit(c)) out += static_cast<char>(c);
    }
    return out;
}

static std::string eval_gsm8k_answer(const std::string & out) {
    size_t pos = out.rfind("####");
    if (pos != std::string::npos) {
        pos += 4;
        while (pos < out.size() && !std::isdigit(static_cast<unsigned char>(out[pos]))) pos++;
        size_t end = pos;
        while (end < out.size() && (std::isdigit(static_cast<unsigned char>(out[end])) || out[end] == ',')) end++;
        if (end > pos) return eval_normalize(out.substr(pos, end - pos));
    }
    for (size_t i = out.size(); i-- > 0;) {
        if (std::isdigit(static_cast<unsigned char>(out[i]))) {
            size_t end = i + 1;
            while (i > 0 && (std::isdigit(static_cast<unsigned char>(out[i - 1])) || out[i - 1] == ',')) i--;
            return eval_normalize(out.substr(i, end - i));
        }
    }
    return "";
}

static std::string eval_mmlu_answer(const std::string & out) {
    size_t pos = out.rfind("Answer:");
    if (pos != std::string::npos) {
        pos += 7;
        while (pos < out.size() && (out[pos] == ' ' || out[pos] == ':' || out[pos] == '\n')) pos++;
        if (pos < out.size() && out[pos] >= 'A' && out[pos] <= 'D') {
            return std::string(1, out[pos]);
        }
    }
    for (const char c : out) {
        if (c >= 'A' && c <= 'D') return std::string(1, c);
    }
    return "";
}

static std::vector<EvalSample> eval_load_gsm8k(const std::string & path, int limit) {
    std::vector<EvalSample> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        try {
            const nlohmann::json j = nlohmann::json::parse(line);
            EvalSample s;
            s.question = j.value("question", std::string());
            s.answer = j.value("answer", std::string());
            if (s.question.empty()) continue;
            out.push_back(std::move(s));
            if (limit > 0 && static_cast<int>(out.size()) >= limit) break;
        } catch (const nlohmann::json::exception &) {}
    }
    return out;
}

static std::vector<EvalSample> eval_load_mmlu(const std::string & path, int limit) {
    std::vector<EvalSample> out;
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> cols = eval_csv_split(line);
        if (cols.size() < 6) continue;
        EvalSample s;
        s.question = cols[0];
        for (int i = 1; i <= 4; i++) s.choices.push_back(cols[static_cast<size_t>(i)]);
        s.answer = cols[5];
        if (s.answer.size() == 1 && s.answer[0] >= 'A' && s.answer[0] <= 'D') {
            out.push_back(std::move(s));
            if (limit > 0 && static_cast<int>(out.size()) >= limit) break;
        }
    }
    return out;
}

static std::vector<EvalSample> eval_load_jsonl(const std::string & path, int limit) {
    std::vector<EvalSample> out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        try {
            const nlohmann::json j = nlohmann::json::parse(line);
            EvalSample s;
            s.question = j.value("question", std::string());
            s.answer = j.value("answer", std::string());
            if (s.question.empty()) continue;
            out.push_back(std::move(s));
            if (limit > 0 && static_cast<int>(out.size()) >= limit) break;
        } catch (const nlohmann::json::exception &) {}
    }
    return out;
}

static int cmd_eval(CliArgs & cli) {
    std::string task = "gsm8k";
    std::string subject = "astronomy";
    std::string data_path;
    std::string out_path;
    int limit = 30;
    for (size_t i = 0; i < cli.sub_args.size(); i++) {
        const std::string & a = cli.sub_args[i];
        if (a == "--task" && i + 1 < cli.sub_args.size()) {
            task = cli.sub_args[++i];
        } else if (a == "--subject" && i + 1 < cli.sub_args.size()) {
            subject = cli.sub_args[++i];
        } else if (a == "--limit" && i + 1 < cli.sub_args.size()) {
            parse_int(cli.sub_args[++i], limit);
        } else if (a == "--data" && i + 1 < cli.sub_args.size()) {
            data_path = cli.sub_args[++i];
        } else if (a == "--out" && i + 1 < cli.sub_args.size()) {
            out_path = cli.sub_args[++i];
        } else if ((a == "--ngl" || a == "--n-gpu-layers") && i + 1 < cli.sub_args.size()) {
            parse_int(cli.sub_args[++i], cli.ngl);
        } else if (a == "--ctx" && i + 1 < cli.sub_args.size()) {
            parse_int(cli.sub_args[++i], cli.n_ctx);
        } else if (!a.empty() && a[0] != '-' && cli.model.empty()) {
            cli.model = a;
        } else {
            fprintf(stderr, "eval: unknown option '%s'\n", a.c_str());
            return 1;
        }
    }
    if (cli.model.empty()) {
        fprintf(stderr, "usage: anvil eval <model> [--task gsm8k|mmlu5|jsonl] [--limit N] [--subject <name>] [--data <file.jsonl>] [--out <report.json>]\n");
        return 1;
    }
    std::string path, friendly;
    if (!resolve_model_arg(cli.model, path, friendly)) {
        fprintf(stderr, "\033[31merror: '%s' is not a file and not a registered model\033[0m\n",
                cli.model.c_str());
        return 1;
    }
    if (!validate_gguf(path)) {
        fprintf(stderr, "\033[31merror: %s\033[0m\n", gguf_check_error(path).c_str());
        return 1;
    }
    cli.model = path;

    std::vector<EvalSample> samples;
    std::vector<EvalSample> shots;
    if (task == "gsm8k") {
        if (!data_path.empty()) {
            samples = eval_load_gsm8k(expand_home(data_path), limit);
        } else {
            samples = eval_fetch_rows("openai/gsm8k", "main", "test", limit);
        }
    } else if (task == "mmlu5") {
        if (!data_path.empty()) {
            samples = eval_load_mmlu(expand_home(data_path), limit);
        } else {
            samples = eval_fetch_rows("cais/mmlu", "all", "test", limit, subject);
            shots = eval_fetch_rows("cais/mmlu", "all", "dev", 5, subject);
            if (shots.size() > 5) shots.resize(5);
        }
    } else if (task == "jsonl") {
        if (data_path.empty()) {
            fprintf(stderr, "error: --data <file.jsonl> required for task 'jsonl'\n");
            return 1;
        }
        samples = eval_load_jsonl(expand_home(data_path), limit);
    } else {
        fprintf(stderr, "error: unknown task '%s' (gsm8k|mmlu5|jsonl)\n", task.c_str());
        return 1;
    }
    if (samples.empty()) {
        fprintf(stderr, "\033[31merror: no samples loaded for task '%s'\033[0m\n", task.c_str());
        return 1;
    }

    const HWInfo hw = probe_hw();
    const ModelMeta meta = read_model_meta(path);
    const int max_ctx = static_cast<int>(meta.trained_ctx);
    AnvilConfig cfg = config_exists() ? load_config() : AnvilConfig{};
    if (cli.ngl >= 0) cfg.ngl = cli.ngl;
    else if (cfg.ngl < 0) cfg.ngl = derive_ngl(hw);
    if (cli.n_ctx > 0) cfg.n_ctx = cli.n_ctx;
    if (cfg.n_ctx <= 0) cfg.n_ctx = max_ctx > 0 ? max_ctx : 2048;

    LlamaBackend backend;
    fprintf(stderr, "Loading model: %s ...\n", path.c_str());
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.ngl;
    LlamaModel model(llama_model_load_from_file(path.c_str(), mparams));
    if (!model) {
        fprintf(stderr, "\033[31merror: failed to load model '%s'\033[0m\n", path.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(cfg.n_ctx);
    cparams.n_batch = static_cast<uint32_t>(std::min<int>(cfg.n_ctx, 8192));
    cparams.n_threads = cfg.n_threads > 0 ? cfg.n_threads : hw.cpu_threads;
    cparams.flash_attn_type = cfg.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k = cfg.type_k;
    cparams.type_v = cfg.type_v;
    LlamaContext ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        fprintf(stderr, "\033[31merror: failed to create context\033[0m\n");
        return 1;
    }
    llama_memory_t mem = llama_get_memory(ctx);
    AnvilConfig eval_cfg = cfg;
    eval_cfg.temp = 0.0f;
    LlamaSampler smpl(build_sampler_chain(vocab, eval_cfg, false, "", 0));
    if (!smpl) {
        fprintf(stderr, "\033[31merror: sampler init failed\033[0m\n");
        return 1;
    }

    const bool is_mmlu = task == "mmlu5";
    int correct = 0;
    const auto t0 = std::chrono::steady_clock::now();
    printf("\n\033[1;36m── anvil eval ──\033[0m\n");
    printf("  task    : %s%s%s\n", task.c_str(), is_mmlu ? " (subject: " : "",
           is_mmlu ? subject.c_str() : "");
    printf("  samples : %zu | limit %d | greedy\n\n", samples.size(), limit);
    printf("%-5s %-48s %-8s %-8s\n", "#", "QUESTION", "PRED", "GOLD");

    for (size_t i = 0; i < samples.size(); i++) {
        const EvalSample & s = samples[i];
        std::string prompt;
        if (is_mmlu) {
            prompt = "The following are multiple choice questions about " + subject +
                     ". Answer with the single correct letter A, B, C, or D.\n\n";
            for (const auto & sh : shots) {
                prompt += "Q: " + sh.question + "\n";
                for (int c = 0; c < 4; c++) {
                    prompt += std::string(1, static_cast<char>('A' + c)) + ". " + sh.choices[static_cast<size_t>(c)] + "\n";
                }
                prompt += "Answer: " + sh.answer + "\n\n";
            }
            prompt += "Q: " + s.question + "\n";
            for (int c = 0; c < 4; c++) {
                prompt += std::string(1, static_cast<char>('A' + c)) + ". " + s.choices[static_cast<size_t>(c)] + "\n";
            }
            prompt += "Answer:";
        } else {
            prompt = "Solve the math problem below step by step, then write the final numeric answer after '####'.\n\nQuestion: " +
                     s.question + "\nAnswer:";
        }
        std::vector<llama_token> toks = tokenize_render(vocab, prompt);
        if (toks.empty()) {
            printf("%-5zu %-48s \033[31mtok fail\033[0m\n", i + 1, "");
            continue;
        }
        std::vector<llama_token> full;
        full.reserve(toks.size() + 1);
        if (llama_vocab_get_add_bos(vocab)) full.push_back(llama_vocab_bos(vocab));
        full.insert(full.end(), toks.begin(), toks.end());
        llama_memory_clear(mem, true);
        const int32_t chunk = static_cast<int32_t>(llama_n_batch(ctx));
        bool ok = true;
        for (size_t k = 0; k < full.size(); k += static_cast<size_t>(chunk)) {
            const size_t nn = std::min<size_t>(static_cast<size_t>(chunk), full.size() - k);
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(full.data() + k),
                                                    static_cast<int32_t>(nn));
            if (llama_decode(ctx, batch) != 0) { ok = false; break; }
        }
        if (!ok) continue;
        llama_sampler_reset(smpl.get());
        std::string gen;
        for (int n = 0; n < 512; n++) {
            const int32_t used = llama_memory_seq_pos_max(mem, 0) + 1;
            if (used >= static_cast<int32_t>(llama_n_ctx(ctx))) break;
            const llama_token id = llama_sampler_sample(smpl.get(), ctx, -1);
            if (llama_vocab_is_eog(vocab, id)) break;
            gen += token_to_str(vocab, id);
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(&id), 1);
            if (llama_decode(ctx, batch) != 0) break;
        }
        const std::string pred = is_mmlu ? eval_mmlu_answer(gen) : eval_gsm8k_answer(gen);
        std::string gold = is_mmlu ? s.answer : eval_gsm8k_answer(s.answer);
        if (gold.empty()) gold = eval_normalize(s.answer);
        const bool hit = !pred.empty() && pred == gold;
        if (hit) correct++;
        std::string q = s.question;
        if (q.size() > 46) q = q.substr(0, 46) + "...";
        const std::string disp = (hit ? "\033[32m" : "\033[31m") +
                                 (pred.empty() ? "-" : pred) + "\033[0m";
        printf("%-5zu %-48s %-8s %-8s\n", i + 1, q.c_str(), disp.c_str(), gold.c_str());
        fflush(stdout);
        if (static_cast<int>(i) % 10 == 9) {
            fprintf(stderr, "\r  %zu/%zu done (%d correct)", i + 1, samples.size(), correct);
            fflush(stderr);
        }
    }
    fprintf(stderr, "\r\033[2K");
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const double acc = samples.empty() ? 0.0 : 100.0 * correct / static_cast<double>(samples.size());
    printf("\n\033[1;36m── Results ──\033[0m\n");
    printf("  accuracy : \033[1m%.2f%%\033[0m (%d/%zu)\n", acc, correct, samples.size());
    printf("  time     : %.1fs\n\n", elapsed);

    if (!out_path.empty()) {
        nlohmann::json report;
        report["task"] = task;
        report["model"] = path;
        report["samples"] = samples.size();
        report["correct"] = correct;
        report["accuracy"] = std::round(acc * 100.0) / 100.0;
        report["elapsed_sec"] = std::round(elapsed * 10.0) / 10.0;
        std::ofstream f(out_path, std::ios::trunc);
        if (f) {
            f << report.dump(2) << "\n";
            printf("  report   : %s\n\n", out_path.c_str());
        }
    }
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    install_sigint(anvil_signal_handler);
    llama_log_set([](enum ggml_log_level level, const char * text, void *) {
        if (level >= GGML_LOG_LEVEL_WARN) fprintf(stderr, "%s", text);
    }, nullptr);

    CliArgs cli = parse_args(argc, argv);
    if (cli.invalid) { print_usage(); return 1; }
    if (cli.help)    { print_usage(); return 0; }
    if (cli.version) { printf("anvil %s\n", ANVIL_VERSION); return 0; }

    if (cli.sub == "models")  return cmd_models(cli.sub_args);
    if (cli.sub == "profile") return cmd_profile(cli.sub_args);
    if (cli.sub == "rm")      return cmd_rm(cli.sub_args);
    if (cli.sub == "pull")    return cmd_pull(cli.sub_args);
    if (cli.sub == "serve")   return cmd_serve(cli);
    if (cli.sub == "bench")   return cmd_bench(cli);
    if (cli.sub == "doctor")  return cmd_doctor(cli.sub_args);
    if (cli.sub == "self-update") return cmd_self_update(cli.sub_args);
    if (cli.sub == "mcp")     return cmd_mcp(cli.sub_args);
    if (cli.sub == "keys")    return cmd_keys(cli.sub_args);
    if (cli.sub == "eval")    return cmd_eval(cli);

    if (cli.model.rfind("gpt:", 0) == 0 || cli.model.rfind("claude:", 0) == 0) {
        return cmd_remote(cli);
    }

    if (cli.model.empty() && !cli.setup) {
        fprintf(stderr, "error: no model specified\n\n");
        print_usage();
        return 1;
    }

    if (!cli.token.empty()) g_hf_token = cli.token;
    const char * hf_env = std::getenv("HF_TOKEN");
    if (g_hf_token.empty() && hf_env && hf_env[0]) g_hf_token = hf_env;

    std::string resolved_path;
    std::string friendly;
    if (!cli.model.empty()) {
        if (!resolve_model_arg(cli.model, resolved_path, friendly)) {
            if (!maybe_auto_pull(cli.model, resolved_path, friendly)) {
                fprintf(stderr, "\033[31merror: '%s' is not a file and not a registered model\033[0m\n",
                        cli.model.c_str());
                fprintf(stderr, "  Try: anvil pull ollama:%s | anvil pull hf:<repo> | anvil models import <file.gguf> --name %s\n",
                        cli.model.c_str(), cli.model.c_str());
                return 1;
            }
        }
        cli.model = resolved_path;
        if (friendly.empty()) {

            friendly = slugify(std::filesystem::path(cli.model).stem().string());
        }
        cli.friendly = friendly;

        if (!validate_gguf(cli.model)) {
            fprintf(stderr, "\033[31merror: %s\033[0m\n", gguf_check_error(cli.model).c_str());
            return 1;
        }
    }

    const HWInfo hw = probe_hw();

    ModelMeta meta;
    if (!cli.model.empty()) meta = read_model_meta(cli.model);
    const int max_ctx = static_cast<int>(meta.trained_ctx);

    LlamaBackend backend;

    AnvilConfig cfg;
    bool setup_ran = false;
    if (cli.setup || !config_exists()) {
        setup_ran = true;
        cfg = run_setup_tui(hw, max_ctx);
        cfg.model = cli.model;
        write_config(cfg);
        fprintf(stderr, "\nConfig saved to %s\n", config_path().c_str());
        if (cli.model.empty()) {

            fprintf(stderr, "Hardware probe complete. Run 'anvil <model>' to start a session.\n");
            return 0;
        }
    } else {
        cfg = load_config();
    }

    fprintf(stderr, "Hardware: %s | %s | %" PRIu64 " GB RAM | %d threads\n",
            hw.cpu.c_str(), hw.arch.c_str(),
            static_cast<uint64_t>(hw.ram_bytes / (1024ULL * 1024 * 1024)),
            hw.cpu_threads);
    for (const auto & gpu : hw.gpus) {
        fprintf(stderr, "GPU: %s (%s) %" PRIu64 " MB VRAM%s\n",
                gpu.name.c_str(), gpu.vendor.c_str(),
                gpu.vram_mb,
                gpu.is_discrete ? " [discrete]" : "");
    }

    std::vector<ModelEntry> models = load_models();
    ModelEntry * entry = find_model(models, friendly);
    if (!entry) {
        entry = auto_register(models, cli.model, friendly, meta);
        if (!save_models(models)) return 1;
        fprintf(stderr, "Registered local model as '%s' (see: anvil models)\n", friendly.c_str());
    }
    if (entry) {

        if (cli.friendly != entry->name) cli.friendly = entry->name;
        friendly = entry->name;
        if (!std::filesystem::exists(entry->path)) {
            fprintf(stderr, "\033[33mwarning: registered model file missing: %s\033[0m\n", entry->path.c_str());
        }

        entry->desc = meta.desc;
        entry->trained_ctx = meta.trained_ctx;
        if (!meta.gguf_meta.empty()) entry->gguf_meta = meta.gguf_meta;
        std::error_code ec;
        const auto sz = std::filesystem::file_size(cli.model, ec);
        if (!ec) entry->size_bytes = sz;

        if (setup_ran) {
            ModelProfile & p = entry->profile;
            p.set("n_ctx", cfg.n_ctx);
            p.set("ngl", cfg.ngl);
            p.set("temp", cfg.temp);
            p.set("flash_attn", cfg.flash_attn);
            p.set("type_k", kv_type_short(cfg.type_k));
            p.set("type_v", kv_type_short(cfg.type_v));
            save_models(models);
            fprintf(stderr, "Setup saved to profile '%s' (%zu settings)\n",
                    friendly.c_str(), p.settings.size());
        }
        entry->profile.apply_to(cfg);
        fprintf(stderr, "Profile '%s': %d setting(s) applied\n", friendly.c_str(),
                static_cast<int>(entry->profile.settings.size()));
    }

    if (!cli.file_prompt.empty()) {
        std::ifstream pf(cli.file_prompt);
        if (!pf) {
            fprintf(stderr, "\033[31merror: cannot open prompt file '%s'\033[0m\n", cli.file_prompt.c_str());
            return 1;
        }
        cli.prompt.assign(std::istreambuf_iterator<char>(pf), std::istreambuf_iterator<char>());
    }

    if (cli.n_ctx > 0)          cfg.n_ctx = cli.n_ctx;
    if (cfg.n_ctx <= 0 && max_ctx > 0) cfg.n_ctx = max_ctx;
    if (cli.ngl >= 0)           cfg.ngl = cli.ngl;
    else if (cfg.ngl < 0)       cfg.ngl = derive_ngl(hw);
    if (cli.temp >= 0)          cfg.temp = cli.temp;
    if (cli.top_k >= 0)         cfg.top_k = cli.top_k;
    if (cli.top_p >= 0)         cfg.top_p = cli.top_p;
    if (cli.repeat_penalty >= 0) cfg.repeat_penalty = cli.repeat_penalty;
    if (cli.n_threads > 0)      cfg.n_threads = cli.n_threads;
    if (cli.flash_attn)         cfg.flash_attn = true;
    if (cli.no_flash_attn)      cfg.flash_attn = false;
    if (cli.mtp)                cfg.mtp = true;
    if (!cli.type_k.empty())    cfg.type_k = kv_type_from_name(cli.type_k);
    if (!cli.type_v.empty())    cfg.type_v = kv_type_from_name(cli.type_v);
    if (cli.seed >= 0)          cfg.seed = cli.seed;
    if (cli.n_batch > 0)        cfg.n_batch = cli.n_batch;
    if (cli.min_p >= 0)         cfg.min_p = cli.min_p;
    if (cli.repeat_last_n >= 0) cfg.repeat_last_n = cli.repeat_last_n;
    if (cli.typical >= 0)       cfg.typical = cli.typical;
    if (cli.mirostat >= 0)      cfg.mirostat = cli.mirostat;
    if (cli.mirostat_lr >= 0)   cfg.mirostat_lr = cli.mirostat_lr;
    if (cli.mirostat_ent >= 0)  cfg.mirostat_ent = cli.mirostat_ent;
    if (cli.ignore_eos)         cfg.ignore_eos = true;
    if (!cli.samplers.empty())  cfg.samplers = cli.samplers;

    if (cli.system_prompt.size() > 1 && cli.system_prompt[0] == '@' && !load_preset_text(cli.system_prompt).empty()) {
        cfg.system_prompt = load_preset_text(cli.system_prompt);
    } else if (!cli.system_prompt.empty()) {
        cfg.system_prompt = cli.system_prompt;
    }

    if (cli.save_profile && entry) {
        ModelProfile & p = entry->profile;
        if (cli.n_ctx > 0)               p.set("n_ctx", cli.n_ctx);
        if (cli.ngl >= 0)                p.set("ngl", cli.ngl);
        if (cli.n_threads > 0)           p.set("n_threads", cli.n_threads);
        if (cli.temp >= 0)               p.set("temp", cli.temp);
        if (cli.top_k >= 0)              p.set("top_k", cli.top_k);
        if (cli.top_p >= 0)              p.set("top_p", cli.top_p);
        if (cli.repeat_penalty >= 0)     p.set("repeat_penalty", cli.repeat_penalty);
        if (cli.flash_attn)              p.set("flash_attn", true);
        if (cli.no_flash_attn)           p.set("flash_attn", false);
        if (cli.mtp)                     p.set("mtp", true);

        if (!cli.type_k.empty() && valid_kv_name(cli.type_k)) p.set("type_k", cli.type_k);
        if (!cli.type_v.empty() && valid_kv_name(cli.type_v)) p.set("type_v", cli.type_v);
        if (cli.seed >= 0)               p.set("seed", cli.seed);
        if (cli.min_p >= 0)              p.set("min_p", cli.min_p);
        if (cli.repeat_last_n >= 0)      p.set("repeat_last_n", cli.repeat_last_n);
        if (cli.typical >= 0)            p.set("typical", cli.typical);
        if (cli.mirostat >= 0)           p.set("mirostat", cli.mirostat);
        if (cli.mirostat_lr >= 0)        p.set("mirostat_lr", cli.mirostat_lr);
        if (cli.mirostat_ent >= 0)       p.set("mirostat_ent", cli.mirostat_ent);
        if (cli.ignore_eos)              p.set("ignore_eos", true);
        if (cli.n_batch > 0)             p.set("n_batch", cli.n_batch);
        if (!cli.samplers.empty())       p.set("samplers", cli.samplers);
        if (!cli.system_prompt.empty())  p.set("system_prompt", cli.system_prompt);
        save_models(models);
        fprintf(stderr, "Saved %zu setting(s) to profile '%s'\n", p.settings.size(), friendly.c_str());
    } else if (entry) {
        save_models(models);
    }

    cfg.model = cli.model;
    if (cfg.system_prompt.empty() && !cli.system_prompt.empty()) {
        cfg.system_prompt = cli.system_prompt;
    }
    write_config(cfg);

    const int rc = run_chat(cli, cfg, hw, models);
    return rc;
}
