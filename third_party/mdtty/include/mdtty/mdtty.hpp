#pragma once

#include <array>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace mdtty {

/// Style configuration for the renderer. All style fields are ANSI escape
/// sequences; override any of them with alternate codes or empty strings.
struct Config {
  const char *bold        = "\033[1m";
  const char *italic      = "\033[3m";
  const char *code_inline = "\033[38;5;220m";
  const char *code_block  = "\033[38;5;114m";
  /// Bold + colored heading styles, graded from h1 (brightest) to h6 (dimmest).
  /// Uses 256-color palette: bright magenta → blue → cyan → green → yellow → dim.
  std::array<const char *, 6> heading = {{
      "\033[1m\033[38;5;199m", // h1 — bold bright magenta
      "\033[1m\033[38;5;39m",  // h2 — bold bright blue
      "\033[1m\033[38;5;49m",  // h3 — bold bright cyan-green
      "\033[1m\033[38;5;114m", // h4 — bold green
      "\033[1m\033[38;5;220m", // h5 — bold yellow
      "\033[38;5;245m",        // h6 — dim gray (no bold)
  }};
  const char *quote       = "\033[2m\033[3m";
  const char *reset       = "\033[0m";
  const char *hr          = "\033[2m";
  const char *table       = "\033[2m";
  const char *table_head  = "\033[1m";

  /// When true, no ANSI escapes are emitted. Auto-enabled in the constructor
  /// if the sink does not target a TTY.
  bool strip_ansi = false;

  /// Terminal width used for horizontal rules. 0 = auto-detect (ioctl on
  /// POSIX, GetConsoleScreenBufferInfo on Windows, fallback 80).
  int terminal_width = 0;
};

/// Streaming markdown → ANSI terminal renderer.
///
/// Characters are fed in arbitrary-sized chunks via feed() and accumulated in
/// an internal line buffer until a newline is seen; only complete lines are
/// classified and emitted. This makes the renderer safe against any chunk
/// boundary — mid-word, mid-span, mid-line — without producing torn output.
///
/// After the last chunk, call flush() to drain the buffer and close any open
/// fenced code block. Reuse across multiple streams via reset().
class Renderer {
public:
  using Sink = std::function<void(std::string_view)>;

  /// Constructs a renderer. If cfg.strip_ansi is false and the sink does not
  /// target a TTY, strip_ansi is force-enabled so piped output is never
  /// polluted with escape codes.
  explicit Renderer(Sink sink, Config cfg = {});

  /// Feeds a chunk of markdown. Safe to call with chunks that split anywhere.
  void feed(std::string_view chunk);

  /// Drains any buffered partial line and closes open state. Idempotent.
  void flush();

  /// Clears line buffer, fence state, and any open inline span. Preserves
  /// sink, config, and cached terminal width so the renderer can be reused.
  void reset();

  /// One-shot helper: renders markdown to a string by internally running the
  /// same feed()+flush() path, so output is byte-identical to streaming.
  static std::string render(std::string_view markdown, Config cfg = {});

  /// Returns true if stdout is currently a TTY.
  static bool is_tty();

private:
  Sink        sink_;
  Config      cfg_;
  std::string line_buf_;
  bool        in_fence_         = false;
  bool        fence_just_opened_ = false;
  int         cached_width_     = 0;
  std::vector<std::string> table_buf_;

  void process_line(std::string_view line);
  void flush_table();
  void emit_inline(std::string_view line);
  void emit_raw(std::string_view s);
  void emit_style(const char *code);
  int  terminal_width();
};

} // namespace mdtty
