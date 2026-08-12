#pragma once

#include <array>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace mdtty {

struct Config {
  const char *bold        = "\033[1m";
  const char *italic      = "\033[3m";
  const char *code_inline = "\033[38;5;220m";
  const char *code_block  = "\033[38;5;114m";

  std::array<const char *, 6> heading = {{
      "\033[1m\033[38;5;199m",
      "\033[1m\033[38;5;39m",
      "\033[1m\033[38;5;49m",
      "\033[1m\033[38;5;114m",
      "\033[1m\033[38;5;220m",
      "\033[38;5;245m",
  }};
  const char *quote       = "\033[2m\033[3m";
  const char *reset       = "\033[0m";
  const char *hr          = "\033[2m";
  const char *table       = "\033[2m";
  const char *table_head  = "\033[1m";

  bool strip_ansi = false;

  int terminal_width = 0;
};

class Renderer {
public:
  using Sink = std::function<void(std::string_view)>;

  explicit Renderer(Sink sink, Config cfg = {});

  void feed(std::string_view chunk);

  void flush();

  void reset();

  static std::string render(std::string_view markdown, Config cfg = {});

  static bool is_tty();

private:
  enum class Span : unsigned char { None, Bold, Italic, Code };

  struct LineDecision {
    enum class Kind : unsigned char { None, Plain, Heading, Quote, Bullet, Numbered } kind = Kind::None;
    std::size_t body_at = 0;
    std::size_t indent  = 0;
    int         level   = 0;
  };

  Sink        sink_;
  Config      cfg_;
  std::string line_buf_;
  std::string out_buf_;
  bool        in_fence_          = false;
  bool        fence_just_opened_ = false;
  int         cached_width_      = 0;
  std::vector<std::string> table_buf_;

  enum class LineClass : unsigned char { Ambiguous, Streaming };
  LineClass    line_class_ = LineClass::Ambiguous;
  LineDecision line_dec_;
  Span         span_           = Span::None;
  char         inline_pending_ = 0;

  void process_line(std::string_view line);
  void flush_table();
  void emit_inline(std::string_view line);
  void emit_raw(std::string_view s);
  void emit_style(const char * code);
  int  terminal_width();

  void flush_out();
  void finish_line(bool allow_empty);
  LineDecision classify_line(const std::string & b);
  void commit_line(const std::string & b, const LineDecision & d);
  void inline_feed_char(char c);
  void inline_flush();
  void inline_open(Span s, const char * code);
  void inline_close();
};

}
