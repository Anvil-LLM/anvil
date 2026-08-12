#include "mdtty/mdtty.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <ranges>
#include <string>
#include <string_view>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include "widechar_width.h"
#pragma GCC diagnostic pop
#include <utf8proc.h>
#include <utility>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <io.h>
#  include <windows.h>
#else
#  include <sys/ioctl.h>
#  include <unistd.h>
#endif

namespace mdtty {

namespace {

constexpr std::string_view k_fence        = "```";
constexpr std::string_view k_hr_char      = "\xe2\x94\x80";
constexpr std::string_view k_quote_gutter = "\xe2\x94\x82 ";

constexpr std::string_view k_tbl_h  = "\xe2\x94\x80";
constexpr std::string_view k_tbl_v  = "\xe2\x94\x82";
constexpr std::string_view k_tbl_tl = "\xe2\x94\x8c";
constexpr std::string_view k_tbl_tr = "\xe2\x94\x90";
constexpr std::string_view k_tbl_bl = "\xe2\x94\x94";
constexpr std::string_view k_tbl_br = "\xe2\x94\x98";
constexpr std::string_view k_tbl_td = "\xe2\x94\xac";
constexpr std::string_view k_tbl_tu = "\xe2\x94\xb4";
constexpr std::string_view k_tbl_lj = "\xe2\x94\x9c";
constexpr std::string_view k_tbl_rj = "\xe2\x94\xa4";
constexpr std::string_view k_tbl_cr = "\xe2\x94\xbc";

constexpr std::array<std::string_view, 3> k_bullets = {
    "\xe2\x80\xa2",
    "\xe2\x97\xa6",
    "\xe2\x96\xb8",
};

std::size_t count_leading_spaces(std::string_view line) {
  std::size_t n = 0;
  while (n < line.size() && line[n] == ' ') {
    ++n;
  }
  return n;
}

std::string_view trim_right(std::string_view s) {
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) {
    s.remove_suffix(1);
  }
  return s;
}

bool is_hr_line(std::string_view line) {
  line = trim_right(line);
  auto content = line;
  while (!content.empty() && content.front() == ' ') {
    content.remove_prefix(1);
  }
  if (content.size() < 3) {
    return false;
  }
  const char marker = content.front();
  if (marker != '-' && marker != '=' && marker != '*') {
    return false;
  }
  std::size_t count = 0;
  const bool uniform = std::ranges::all_of(content, [&](char c) {
    if (c == ' ') {
      return true;
    }
    if (c == marker) {
      ++count;
      return true;
    }
    return false;
  });
  return uniform && count >= 3;
}

bool is_fence_line(std::string_view line) {
  return trim_right(line).starts_with(k_fence);
}

std::string_view fence_lang(std::string_view line) {
  auto t = trim_right(line);
  t.remove_prefix(k_fence.size());
  while (!t.empty() && t.front() == ' ') {
    t.remove_prefix(1);
  }
  return t;
}

int detect_terminal_width() {
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO csbi{};
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi) != 0) {
    const int w = csbi.srWindow.Right - csbi.srWindow.Left + 1;
    if (w > 0) {
      return w;
    }
  }
#else
  struct winsize ws {};
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
    return static_cast<int>(ws.ws_col);
  }
#endif
  return 80;
}

bool is_table_line(std::string_view line) {
  while (!line.empty() && line.front() == ' ') {
    line.remove_prefix(1);
  }
  return !line.empty() && line.front() == '|';
}

std::string_view trim_cell(std::string_view cell) {
  while (!cell.empty() && cell.front() == ' ') {
    cell.remove_prefix(1);
  }
  while (!cell.empty() && cell.back() == ' ') {
    cell.remove_suffix(1);
  }
  return cell;
}

std::vector<std::string_view> split_table_row(std::string_view line) {
  line = trim_right(line);
  if (!line.empty() && line.front() == '|') {
    line.remove_prefix(1);
  }
  if (!line.empty() && line.back() == '|') {
    line.remove_suffix(1);
  }
  std::vector<std::string_view> cells;
  std::size_t start = 0;
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] == '|') {
      cells.push_back(trim_cell(line.substr(start, i - start)));
      start = i + 1;
    }
  }
  cells.push_back(trim_cell(line.substr(start)));
  return cells;
}

bool is_separator_cell(std::string_view cell) {
  if (cell.empty()) {
    return false;
  }
  if (cell.front() == ':') {
    cell.remove_prefix(1);
  }
  if (!cell.empty() && cell.back() == ':') {
    cell.remove_suffix(1);
  }
  if (cell.size() < 1) {
    return false;
  }
  return std::ranges::all_of(cell, [](char c) { return c == '-'; });
}

bool is_text_presentation_wide_emoji(uint32_t cp) {

  switch (cp) {
  case 0x203C: case 0x2049: case 0x2122: case 0x2139:
  case 0x2194: case 0x2195: case 0x2196: case 0x2197: case 0x2198: case 0x2199:
  case 0x21A9: case 0x21AA:
  case 0x2328: case 0x23CF:
  case 0x24C2:
  case 0x25AA: case 0x25AB: case 0x25B6: case 0x25C0:
  case 0x25FB: case 0x25FC:
  case 0x2600: case 0x2601: case 0x2602: case 0x2603: case 0x2604:
  case 0x260E: case 0x2611:
  case 0x2618: case 0x261D: case 0x2620:
  case 0x2622: case 0x2623: case 0x2626: case 0x262A:
  case 0x262E: case 0x262F:
  case 0x2638: case 0x2639: case 0x263A:
  case 0x2640: case 0x2642:
  case 0x265F: case 0x2660: case 0x2663: case 0x2665: case 0x2666: case 0x2668:
  case 0x267B: case 0x267E:
  case 0x2692: case 0x2694: case 0x2695: case 0x2696: case 0x2697:
  case 0x2699: case 0x269B: case 0x269C:
  case 0x26A0: case 0x26A7:
  case 0x26B0: case 0x26B1:
  case 0x26C8: case 0x26CF: case 0x26D1:
  case 0x26D3: case 0x26E9:
  case 0x26F0: case 0x26F1: case 0x26F4: case 0x26F7: case 0x26F8: case 0x26F9:
  case 0x2702: case 0x2708: case 0x2709:
  case 0x270C: case 0x270D: case 0x270F: case 0x2712:
  case 0x2714: case 0x2716: case 0x271D: case 0x2721:
  case 0x2733: case 0x2734: case 0x2744: case 0x2747:
  case 0x2763: case 0x2764: case 0x27A1:
  case 0x2934: case 0x2935:
  case 0x2B05: case 0x2B06: case 0x2B07:
  case 0x3030: case 0x303D: case 0x3297: case 0x3299:
    return true;
  default:
    return false;
  }
}

int terminal_charwidth(utf8proc_int32_t cp) {
  if (cp < 0)
    return 1;

  if (is_text_presentation_wide_emoji(static_cast<uint32_t>(cp)))
    return 2;
  int w = widechar_wcwidth(static_cast<uint32_t>(cp));
  if (w == widechar_combining)
    return 0;

  if (w == widechar_widened_in_9)
    return 2;

  if (w < 0)
    return 1;
  return w;
}

void add_codepoint_width(std::string_view text, std::size_t &i, std::size_t &w,
                          utf8proc_int32_t &prev_cp, utf8proc_int32_t &state) {
  utf8proc_int32_t cp = 0;
  auto consumed = utf8proc_iterate(
      reinterpret_cast<const utf8proc_uint8_t *>(text.data() + i),
      static_cast<utf8proc_ssize_t>(text.size() - i), &cp);
  if (consumed < 1) {

    ++w;
    ++i;
    return;
  }
  if (prev_cp != 0 && !utf8proc_grapheme_break_stateful(prev_cp, cp, &state)) {

  } else {
    int cw = terminal_charwidth(cp);
    if (cw > 0)
      w += static_cast<std::size_t>(cw);
  }
  prev_cp = cp;
  i += static_cast<std::size_t>(consumed);
}

std::size_t visual_width(std::string_view text) {
  std::size_t w = 0;
  std::size_t i = 0;
  bool in_code = false;
  utf8proc_int32_t prev_cp = 0;
  utf8proc_int32_t state = 0;
  while (i < text.size()) {
    char c = text[i];
    if (in_code) {
      if (c == '`') {
        in_code = false;
        ++i;
        prev_cp = 0;
        state = 0;
        continue;
      }
      add_codepoint_width(text, i, w, prev_cp, state);
      continue;
    }
    if (c == '\\' && i + 1 < text.size()) {
      ++i;
      add_codepoint_width(text, i, w, prev_cp, state);
      continue;
    }
    if (c == '`') {
      in_code = true;
      ++i;
      prev_cp = 0;
      state = 0;
      continue;
    }
    if ((c == '*' || c == '_') && i + 1 < text.size() && text[i + 1] == c) {
      i += 2;
      continue;
    }
    if (c == '*' || c == '_') {
      ++i;
      continue;
    }
    add_codepoint_width(text, i, w, prev_cp, state);
  }
  return w;
}

bool stdout_is_tty() {
#if defined(_WIN32)
  DWORD mode = 0;
  return GetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), &mode) != 0;
#else
  return ::isatty(STDOUT_FILENO) != 0;
#endif
}

}

Renderer::Renderer(Sink sink, Config cfg) : sink_(std::move(sink)), cfg_(cfg) {
  if (!cfg_.strip_ansi && !stdout_is_tty()) {
    cfg_.strip_ansi = true;
  }
}

void Renderer::emit_raw(std::string_view s) {
  if (!s.empty()) {
    out_buf_.append(s.data(), s.size());
  }
}

void Renderer::flush_out() {
  if (!out_buf_.empty()) {
    if (sink_) {
      sink_(out_buf_);
    }
    out_buf_.clear();
  }
}

void Renderer::emit_style(const char * code) {
  if (!cfg_.strip_ansi && code != nullptr && code[0] != '\0') {
    flush_out();
    if (sink_) {
      sink_(std::string_view(code));
    }
  }
}

int Renderer::terminal_width() {
  if (cfg_.terminal_width > 0) {
    return cfg_.terminal_width;
  }
  if (cached_width_ == 0) {
    cached_width_ = detect_terminal_width();
  }
  return cached_width_;
}

void Renderer::feed(std::string_view chunk) {
  for (char c : chunk) {
    if (c == '\n') {
      finish_line(true);
    } else if (line_class_ == LineClass::Streaming) {
      inline_feed_char(c);
    } else {
      line_buf_.push_back(c);
      if (!in_fence_) {
        const LineDecision d = classify_line(line_buf_);
        if (d.kind != LineDecision::Kind::None) {
          commit_line(line_buf_, d);
          line_buf_.clear();
          line_class_ = LineClass::Streaming;
          line_dec_   = d;
        }
      }
    }
  }
  flush_out();
}

void Renderer::flush() {
  finish_line(false);
  flush_table();
  if (in_fence_) {
    emit_style(cfg_.reset);
    in_fence_          = false;
    fence_just_opened_ = false;
  }
  flush_out();
}

void Renderer::reset() {
  line_buf_.clear();
  out_buf_.clear();
  in_fence_          = false;
  fence_just_opened_ = false;
  table_buf_.clear();
  line_class_ = LineClass::Ambiguous;
  line_dec_   = LineDecision{};
  span_       = Span::None;
  inline_pending_ = 0;
}

void Renderer::process_line(std::string_view line) {

  if (!table_buf_.empty() && !in_fence_) {
    if (is_table_line(line)) {
      table_buf_.emplace_back(line);
      return;
    }
    flush_table();

  }

  if (in_fence_) {
    if (is_fence_line(line)) {
      in_fence_         = false;
      fence_just_opened_ = false;
      return;
    }
    emit_style(cfg_.code_block);
    emit_raw(line);
    emit_style(cfg_.reset);
    emit_raw("\n");
    return;
  }

  if (is_fence_line(line)) {
    in_fence_         = true;
    fence_just_opened_ = true;
    auto lang         = fence_lang(line);
    if (!lang.empty()) {
      emit_style(cfg_.hr);
      emit_raw(lang);
      emit_style(cfg_.reset);
      emit_raw("\n");
    }
    return;
  }

  if (trim_right(line).empty()) {
    emit_raw("\n");
    return;
  }

  if (is_hr_line(line)) {
    const int w = terminal_width();
    emit_style(cfg_.hr);
    std::string bar;
    bar.reserve(static_cast<std::size_t>(w) * k_hr_char.size());
    for (int i = 0; i < w; ++i) {
      bar.append(k_hr_char);
    }
    emit_raw(bar);
    emit_style(cfg_.reset);
    emit_raw("\n");
    return;
  }

  if (line.starts_with('#')) {
    std::size_t level = 0;
    while (level < line.size() && level < 6 && line[level] == '#') {
      ++level;
    }
    if (level < line.size() && line[level] == ' ') {
      auto body = line;
      body.remove_prefix(level + 1);
      emit_style(cfg_.heading[std::min(level, std::size_t{6}) - 1]);
      emit_raw(line.substr(0, level + 1));
      emit_inline(body);
      emit_style(cfg_.reset);
      emit_raw("\n");
      return;
    }
  }

  if (line.starts_with('>')) {
    auto body = line;
    body.remove_prefix(1);
    if (!body.empty() && body.front() == ' ') {
      body.remove_prefix(1);
    }
    emit_style(cfg_.quote);
    emit_raw(k_quote_gutter);
    emit_inline(body);
    emit_style(cfg_.reset);
    emit_raw("\n");
    return;
  }

  {
    const std::size_t indent = count_leading_spaces(line);
    auto              rest   = line.substr(indent);
    if (rest.size() >= 2 && (rest[0] == '-' || rest[0] == '*' || rest[0] == '+') &&
        rest[1] == ' ') {
      const std::size_t depth  = indent / 2;
      const auto        bullet = k_bullets[std::min(depth, k_bullets.size() - 1)];
      for (std::size_t i = 0; i < indent; ++i) {
        emit_raw(" ");
      }
      emit_raw(bullet);
      emit_raw(" ");
      emit_inline(rest.substr(2));
      emit_raw("\n");
      return;
    }
  }

  {
    const std::size_t indent = count_leading_spaces(line);
    auto              rest   = line.substr(indent);
    std::size_t       digits = 0;
    while (digits < rest.size() && rest[digits] >= '0' && rest[digits] <= '9') {
      ++digits;
    }
    if (digits > 0 && digits + 1 < rest.size() && rest[digits] == '.' &&
        rest[digits + 1] == ' ') {
      for (std::size_t i = 0; i < indent; ++i) {
        emit_raw(" ");
      }
      emit_raw(rest.substr(0, digits + 2));
      emit_inline(rest.substr(digits + 2));
      emit_raw("\n");
      return;
    }
  }

  if (is_table_line(line)) {
    table_buf_.emplace_back(line);
    return;
  }

  emit_inline(line);
  emit_raw("\n");
}

void Renderer::finish_line(bool allow_empty) {
  if (line_class_ == LineClass::Streaming) {
    inline_flush();
    if (line_dec_.kind == LineDecision::Kind::Heading ||
        line_dec_.kind == LineDecision::Kind::Quote) {
      emit_style(cfg_.reset);
    }
    emit_raw("\n");
  } else if (allow_empty || !line_buf_.empty() || in_fence_) {
    emit_style(cfg_.reset);
    process_line(line_buf_);
  }
  line_buf_.clear();
  line_class_ = LineClass::Ambiguous;
  line_dec_   = LineDecision{};
}

Renderer::LineDecision Renderer::classify_line(const std::string & b) {
  LineDecision d;
  if (b.rfind("```", 0) == 0) {
    return d;
  }
  std::size_t i = 0;
  while (i < b.size() && b[i] == ' ') {
    ++i;
  }
  if (i < b.size() && b[i] == '|') {
    return d;
  }
  std::size_t h = 0;
  while (h < b.size() && h < 6 && b[h] == '#') {
    ++h;
  }
  if (h > 0) {
    if (h == b.size()) {
      return d;
    }
    if (b[h] == ' ') {
      d.kind    = LineDecision::Kind::Heading;
      d.level   = static_cast<int>(h);
      d.body_at = h + 1;
      return d;
    }
    d.kind    = LineDecision::Kind::Plain;
    d.body_at = 0;
    return d;
  }
  if (b[0] == '>') {
    for (std::size_t k = 0; k < b.size(); ++k) {
      if (b[k] != '>') {
        d.kind    = LineDecision::Kind::Quote;
        d.body_at = 1;
        if (b.size() > 1 && b[1] == ' ') {
          d.body_at = 2;
        }
        return d;
      }
    }
    return d;
  }
  std::string_view rest(b.data() + i, b.size() - i);
  if (rest.empty()) {
    return d;
  }
  const char m = rest[0];
  if (m == '-' || m == '*' || m == '+' || m == '=') {
    const bool blocky = std::ranges::all_of(rest, [](char c) {
      return c == '-' || c == '*' || c == '+' || c == '=' || c == ' ';
    });
    if (blocky) {
      return d;
    }
    if (rest.size() >= 2 && rest[1] == ' ') {
      d.kind    = LineDecision::Kind::Bullet;
      d.indent  = i;
      d.body_at = i + 2;
      return d;
    }
    d.kind    = LineDecision::Kind::Plain;
    d.body_at = 0;
    return d;
  }
  std::size_t dig = 0;
  while (dig < rest.size() && rest[dig] >= '0' && rest[dig] <= '9') {
    ++dig;
  }
  if (dig > 0) {
    if (dig == rest.size()) {
      return d;
    }
    if (rest[dig] == '.') {
      if (dig + 1 < rest.size() && rest[dig + 1] == ' ') {
        d.kind    = LineDecision::Kind::Numbered;
        d.body_at = i + dig + 2;
        return d;
      }
      if (dig + 1 == rest.size()) {
        return d;
      }
    }
    d.kind    = LineDecision::Kind::Plain;
    d.body_at = 0;
    return d;
  }
  d.kind    = LineDecision::Kind::Plain;
  d.body_at = 0;
  return d;
}

void Renderer::commit_line(const std::string & b, const LineDecision & d) {
  if (!table_buf_.empty() && !in_fence_) {
    flush_table();
  }
  emit_style(cfg_.reset);
  switch (d.kind) {
    case LineDecision::Kind::Plain:
      for (char c : b) {
        inline_feed_char(c);
      }
      break;
    case LineDecision::Kind::Heading:
      emit_style(cfg_.heading[d.level - 1]);
      emit_raw(b.substr(0, d.body_at));
      for (std::size_t k = d.body_at; k < b.size(); ++k) {
        inline_feed_char(b[k]);
      }
      break;
    case LineDecision::Kind::Quote:
      emit_style(cfg_.quote);
      emit_raw(k_quote_gutter);
      for (std::size_t k = d.body_at; k < b.size(); ++k) {
        inline_feed_char(b[k]);
      }
      break;
    case LineDecision::Kind::Bullet: {
      const std::size_t depth  = d.indent / 2;
      const auto        bullet = k_bullets[std::min(depth, k_bullets.size() - 1)];
      for (std::size_t k = 0; k < d.indent; ++k) {
        emit_raw(" ");
      }
      emit_raw(bullet);
      emit_raw(" ");
      for (std::size_t k = d.body_at; k < b.size(); ++k) {
        inline_feed_char(b[k]);
      }
      break;
    }
    case LineDecision::Kind::Numbered:
      emit_raw(b.substr(0, d.body_at));
      for (std::size_t k = d.body_at; k < b.size(); ++k) {
        inline_feed_char(b[k]);
      }
      break;
    case LineDecision::Kind::None:
      break;
  }
}

void Renderer::flush_table() {
  if (table_buf_.empty()) {
    return;
  }

  bool valid = table_buf_.size() >= 2;
  if (valid) {
    auto sep_cells = split_table_row(table_buf_[1]);
    valid = !sep_cells.empty() &&
            std::ranges::all_of(sep_cells, [](auto c) { return is_separator_cell(c); });
  }

  if (!valid) {

    for (auto &row : table_buf_) {
      emit_inline(row);
      emit_raw("\n");
    }
    table_buf_.clear();
    return;
  }

  std::vector<std::vector<std::string_view>> rows;
  std::size_t num_cols = 0;
  for (std::size_t i = 0; i < table_buf_.size(); ++i) {
    if (i == 1) {
      continue;
    }
    auto cells = split_table_row(table_buf_[i]);
    num_cols   = std::max(num_cols, cells.size());
    rows.push_back(std::move(cells));
  }

  std::vector<std::size_t> widths(num_cols, 0);
  for (auto &row : rows) {
    for (std::size_t c = 0; c < row.size(); ++c) {
      widths[c] = std::max(widths[c], visual_width(row[c]));
    }
  }
  for (auto &w : widths) {
    w = std::max(w, std::size_t{1});
  }

  auto emit_border = [&](std::string_view left, std::string_view mid,
                         std::string_view right) {
    emit_style(cfg_.table);
    emit_raw(left);
    for (std::size_t c = 0; c < num_cols; ++c) {
      for (std::size_t i = 0; i < widths[c] + 2; ++i) {
        emit_raw(k_tbl_h);
      }
      emit_raw(c + 1 < num_cols ? mid : right);
    }
    emit_style(cfg_.reset);
    emit_raw("\n");
  };

  auto emit_row = [&](std::vector<std::string_view> &row, const char *cell_style) {
    for (std::size_t c = 0; c < num_cols; ++c) {
      emit_style(cfg_.table);
      emit_raw(k_tbl_v);
      emit_style(cfg_.reset);
      emit_raw(" ");
      std::string_view content = c < row.size() ? row[c] : std::string_view{};
      if (cell_style != nullptr) {
        emit_style(cell_style);
      }
      emit_inline(content);
      if (cell_style != nullptr) {
        emit_style(cfg_.reset);
      }
      std::size_t vw = visual_width(content);
      for (std::size_t p = vw; p < widths[c]; ++p) {
        emit_raw(" ");
      }
      emit_raw(" ");
    }
    emit_style(cfg_.table);
    emit_raw(k_tbl_v);
    emit_style(cfg_.reset);
    emit_raw("\n");
  };

  emit_border(k_tbl_tl, k_tbl_td, k_tbl_tr);

  emit_row(rows[0], cfg_.table_head);

  emit_border(k_tbl_lj, k_tbl_cr, k_tbl_rj);

  for (std::size_t r = 1; r < rows.size(); ++r) {
    emit_row(rows[r], nullptr);
  }

  emit_border(k_tbl_bl, k_tbl_tu, k_tbl_br);

  table_buf_.clear();
}

void Renderer::emit_inline(std::string_view line) {
  const Span saved_span    = span_;
  const char saved_pending = inline_pending_;
  span_           = Span::None;
  inline_pending_ = 0;
  for (char c : line) {
    inline_feed_char(c);
  }
  inline_flush();
  span_           = saved_span;
  inline_pending_ = saved_pending;
}

void Renderer::inline_open(Span s, const char * code) {
  emit_style(code);
  span_ = s;
}

void Renderer::inline_close() {
  if (span_ != Span::None) {
    emit_style(cfg_.reset);
    span_ = Span::None;
  }
}

void Renderer::inline_feed_char(char c) {
  if (inline_pending_ != 0) {
    const char p = inline_pending_;
    inline_pending_ = 0;
    if (p == '\\') {
      emit_raw(std::string_view(&c, 1));
      return;
    }
    if ((p == '*' || p == '_') && c == p) {
      if (span_ == Span::Bold) {
        inline_close();
      } else if (span_ == Span::None) {
        inline_open(Span::Bold, cfg_.bold);
      } else {
        emit_raw(std::string_view(&p, 1));
        emit_raw(std::string_view(&c, 1));
      }
      return;
    }
    if (p == '*' || p == '_') {
      if (span_ == Span::Italic) {
        inline_close();
      } else if (span_ == Span::None) {
        inline_open(Span::Italic, cfg_.italic);
      } else {
        emit_raw(std::string_view(&p, 1));
      }
    } else {
      emit_raw(std::string_view(&p, 1));
    }
  }
  if (c == '`') {
    if (span_ == Span::None) {
      inline_open(Span::Code, cfg_.code_inline);
    } else if (span_ == Span::Code) {
      inline_close();
    } else {
      emit_raw("`");
    }
    return;
  }
  if (c == '\\' && span_ != Span::Code) {
    inline_pending_ = c;
    return;
  }
  if (c == '*' || c == '_') {
    if (span_ == Span::Code) {
      emit_raw(std::string_view(&c, 1));
    } else {
      inline_pending_ = c;
    }
    return;
  }
  emit_raw(std::string_view(&c, 1));
}

void Renderer::inline_flush() {
  if (inline_pending_ != 0) {
    const char p = inline_pending_;
    inline_pending_ = 0;
    if (p == '\\') {
      emit_raw("\\");
    } else if (p == '*' || p == '_') {
      if (span_ == Span::Italic) {
        inline_close();
      } else if (span_ == Span::None) {
        inline_open(Span::Italic, cfg_.italic);
      } else {
        emit_raw(std::string_view(&p, 1));
      }
    } else {
      emit_raw(std::string_view(&p, 1));
    }
  }
  inline_close();
}

std::string Renderer::render(std::string_view markdown, Config cfg) {
  std::string out;

  Renderer r(std::function<void(std::string_view)>([&out](std::string_view s) { out.append(s); }), cfg);

  r.cfg_.strip_ansi = cfg.strip_ansi;
  r.feed(markdown);
  r.flush();
  return out;
}

bool Renderer::is_tty() {
  return stdout_is_tty();
}

}
