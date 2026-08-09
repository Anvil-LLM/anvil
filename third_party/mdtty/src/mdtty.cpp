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
#  include <io.h>
#  include <windows.h>
#else
#  include <sys/ioctl.h>
#  include <unistd.h>
#endif

namespace mdtty {

namespace {

constexpr std::string_view k_fence        = "```";
constexpr std::string_view k_hr_char      = "\xe2\x94\x80"; // U+2500 ─
constexpr std::string_view k_quote_gutter = "\xe2\x94\x82 "; // U+2502 "│ "

// Table box-drawing characters.
constexpr std::string_view k_tbl_h  = "\xe2\x94\x80"; // ─
constexpr std::string_view k_tbl_v  = "\xe2\x94\x82"; // │
constexpr std::string_view k_tbl_tl = "\xe2\x94\x8c"; // ┌
constexpr std::string_view k_tbl_tr = "\xe2\x94\x90"; // ┐
constexpr std::string_view k_tbl_bl = "\xe2\x94\x94"; // └
constexpr std::string_view k_tbl_br = "\xe2\x94\x98"; // ┘
constexpr std::string_view k_tbl_td = "\xe2\x94\xac"; // ┬
constexpr std::string_view k_tbl_tu = "\xe2\x94\xb4"; // ┴
constexpr std::string_view k_tbl_lj = "\xe2\x94\x9c"; // ├
constexpr std::string_view k_tbl_rj = "\xe2\x94\xa4"; // ┤
constexpr std::string_view k_tbl_cr = "\xe2\x94\xbc"; // ┼

constexpr std::array<std::string_view, 3> k_bullets = {
    "\xe2\x80\xa2", // • U+2022
    "\xe2\x97\xa6", // ◦ U+25E6
    "\xe2\x96\xb8", // ▸ U+25B8
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

/// 3+ uniform - = or * characters (ignoring spaces).
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

/// Returns true if the line looks like a table row (starts with '|').
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

/// Splits a table row on '|', stripping the leading/trailing pipes.
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

/// Returns true if the cell content is a valid separator (e.g. "---", ":---:", ":---").
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

/// Returns the column width of \p cp, overriding utf8proc for emoji codepoints
/// that modern terminals render at width 2 even though their formal Unicode
/// East_Asian_Width is Neutral. Source: emoji-data.txt's Emoji_Presentation=Yes
/// property, grouped into broad ranges. Keep this list in sync with promptty.
/// True if \p cp is an Emoji=Yes / Emoji_Presentation=No codepoint that
/// terminals nevertheless render at width 2. Unicode classifies these as
/// "default text presentation," but in practice modern terminals (kitty,
/// wezterm, ghostty, foot, GNOME Terminal, iTerm2) ignore that and render
/// them as emoji. This list is the canonical set; keep it in sync with
/// promptty's copy.
bool is_text_presentation_wide_emoji(uint32_t cp) {
  // Sorted by codepoint for readability.
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

/// Returns the column width of \p cp using the widecharwidth table, which is
/// generated from the upstream Unicode data files (UnicodeData.txt,
/// EastAsianWidth.txt, emoji-data.txt). This is the same source modern
/// terminals (kitty, wezterm, ghostty, foot, GNOME Terminal, iTerm2) use.
int terminal_charwidth(utf8proc_int32_t cp) {
  if (cp < 0)
    return 1;
  // Default-text emoji that terminals render wide anyway. Checked first
  // because widecharwidth would otherwise report these as width 1.
  if (is_text_presentation_wide_emoji(static_cast<uint32_t>(cp)))
    return 2;
  int w = widechar_wcwidth(static_cast<uint32_t>(cp));
  if (w == widechar_combining)
    return 0;
  // widened_in_9: Unicode 9 promoted these to wide because of emoji
  // presentation. Modern terminals render them at width 2.
  if (w == widechar_widened_in_9)
    return 2;
  // Other negative codes (nonprint/ambiguous/private/unassigned) fall back
  // to width 1: safe default for "printable but we don't know."
  if (w < 0)
    return 1;
  return w;
}

/// Adds the column width of one codepoint at \p text[i] to \p w, advancing \p i
/// past the consumed bytes. Handles grapheme clusters via \p prev_cp / \p state
/// so that combining marks and ZWJ sequences count as a single cluster.
void add_codepoint_width(std::string_view text, std::size_t &i, std::size_t &w,
                          utf8proc_int32_t &prev_cp, utf8proc_int32_t &state) {
  utf8proc_int32_t cp = 0;
  auto consumed = utf8proc_iterate(
      reinterpret_cast<const utf8proc_uint8_t *>(text.data() + i),
      static_cast<utf8proc_ssize_t>(text.size() - i), &cp);
  if (consumed < 1) {
    // Malformed byte: skip one and treat as width 1 to stay roughly aligned.
    ++w;
    ++i;
    return;
  }
  if (prev_cp != 0 && !utf8proc_grapheme_break_stateful(prev_cp, cp, &state)) {
    // Continuation of the previous grapheme cluster — width 0.
  } else {
    int cw = terminal_charwidth(cp);
    if (cw > 0)
      w += static_cast<std::size_t>(cw);
  }
  prev_cp = cp;
  i += static_cast<std::size_t>(consumed);
}

/// Computes the visual width of inline markdown text (strips formatting markers).
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
      ++i; // skip the backslash
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

} // namespace

Renderer::Renderer(Sink sink, Config cfg) : sink_(std::move(sink)), cfg_(cfg) {
  if (!cfg_.strip_ansi && !stdout_is_tty()) {
    cfg_.strip_ansi = true;
  }
}

void Renderer::emit_raw(std::string_view s) {
  if (!s.empty() && sink_) {
    sink_(s);
  }
}

void Renderer::emit_style(const char *code) {
  if (!cfg_.strip_ansi && code != nullptr && code[0] != '\0') {
    sink_(std::string_view(code));
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
      process_line(line_buf_);
      line_buf_.clear();
    } else {
      line_buf_.push_back(c);
    }
  }
}

void Renderer::flush() {
  if (!line_buf_.empty()) {
    process_line(line_buf_);
    line_buf_.clear();
  }
  flush_table();
  if (in_fence_) {
    emit_style(cfg_.reset);
    in_fence_         = false;
    fence_just_opened_ = false;
  }
}

void Renderer::reset() {
  line_buf_.clear();
  in_fence_         = false;
  fence_just_opened_ = false;
  table_buf_.clear();
}

void Renderer::process_line(std::string_view line) {
  // --- Table continuation ---
  if (!table_buf_.empty() && !in_fence_) {
    if (is_table_line(line)) {
      table_buf_.emplace_back(line);
      return;
    }
    flush_table();
    // Fall through to process current line normally.
  }

  // --- Fenced code block handling ---
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

  // --- Blank line ---
  if (trim_right(line).empty()) {
    emit_raw("\n");
    return;
  }

  // --- Horizontal rule ---
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

  // --- Heading ---
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

  // --- Blockquote ---
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

  // --- Unordered list ---
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

  // --- Ordered list ---
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

  // --- Table start ---
  if (is_table_line(line)) {
    table_buf_.emplace_back(line);
    return;
  }

  // --- Paragraph ---
  emit_inline(line);
  emit_raw("\n");
}

void Renderer::flush_table() {
  if (table_buf_.empty()) {
    return;
  }

  // Validate: second row must be a separator (e.g. |---|---|).
  bool valid = table_buf_.size() >= 2;
  if (valid) {
    auto sep_cells = split_table_row(table_buf_[1]);
    valid = !sep_cells.empty() &&
            std::ranges::all_of(sep_cells, [](auto c) { return is_separator_cell(c); });
  }

  if (!valid) {
    // Not a real table, emit buffered lines as paragraphs.
    for (auto &row : table_buf_) {
      emit_inline(row);
      emit_raw("\n");
    }
    table_buf_.clear();
    return;
  }

  // Parse rows into cells (skip the separator at index 1).
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

  // Compute column widths from visual (inline-stripped) content.
  std::vector<std::size_t> widths(num_cols, 0);
  for (auto &row : rows) {
    for (std::size_t c = 0; c < row.size(); ++c) {
      widths[c] = std::max(widths[c], visual_width(row[c]));
    }
  }
  for (auto &w : widths) {
    w = std::max(w, std::size_t{1});
  }

  // Helper: emit a horizontal border line.
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

  // Helper: emit a data row.
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

  // Top border.
  emit_border(k_tbl_tl, k_tbl_td, k_tbl_tr);

  // Header row.
  emit_row(rows[0], cfg_.table_head);

  // Header/body separator.
  emit_border(k_tbl_lj, k_tbl_cr, k_tbl_rj);

  // Body rows.
  for (std::size_t r = 1; r < rows.size(); ++r) {
    emit_row(rows[r], nullptr);
  }

  // Bottom border.
  emit_border(k_tbl_bl, k_tbl_tu, k_tbl_br);

  table_buf_.clear();
}

void Renderer::emit_inline(std::string_view line) {
  enum class Span : unsigned char { None, Bold, Italic, Code };
  Span        span   = Span::None;
  std::string buffer;
  buffer.reserve(line.size());

  auto close_span = [&]() {
    if (!buffer.empty()) {
      emit_raw(buffer);
      buffer.clear();
    }
    if (span != Span::None) {
      emit_style(cfg_.reset);
      span = Span::None;
    }
  };

  auto open_span = [&](Span s, const char *code) {
    if (!buffer.empty()) {
      emit_raw(buffer);
      buffer.clear();
    }
    emit_style(code);
    span = s;
  };

  const std::size_t n = line.size();
  std::size_t       i = 0;
  while (i < n) {
    const char c = line[i];

    if (span == Span::Code) {
      if (c == '`') {
        close_span();
        ++i;
        continue;
      }
      buffer.push_back(c);
      ++i;
      continue;
    }

    if (c == '\\' && i + 1 < n) {
      buffer.push_back(line[i + 1]);
      i += 2;
      continue;
    }

    // Bold: ** or __ (checked before italic).
    if ((c == '*' || c == '_') && i + 1 < n && line[i + 1] == c) {
      if (span == Span::Bold) {
        close_span();
      } else if (span == Span::None) {
        open_span(Span::Bold, cfg_.bold);
      } else {
        buffer.push_back(c);
        buffer.push_back(c);
      }
      i += 2;
      continue;
    }

    // Italic: single * or _.
    if (c == '*' || c == '_') {
      if (span == Span::Italic) {
        close_span();
      } else if (span == Span::None) {
        open_span(Span::Italic, cfg_.italic);
      } else {
        buffer.push_back(c);
      }
      ++i;
      continue;
    }

    // Inline code.
    if (c == '`') {
      if (span == Span::None) {
        open_span(Span::Code, cfg_.code_inline);
      } else {
        buffer.push_back(c);
      }
      ++i;
      continue;
    }

    buffer.push_back(c);
    ++i;
  }

  // Auto-close any open span at end-of-line so ANSI never leaks across lines.
  close_span();
}

std::string Renderer::render(std::string_view markdown, Config cfg) {
  std::string out;
  // Force strip_ansi based on the caller's choice, not the TTY state of this
  // process: the static helper returns a string, it does not write to stdout.
  // The constructor's auto-detect would otherwise strip ANSI when stdout is
  // not a TTY even though the caller asked for a styled string.
  // `auto(...)` (decay-copy, C++23) replaced with an explicit std::function
  // so the whole library builds as C++20 — the only C++23 construct.
  Renderer r(std::function<void(std::string_view)>([&out](std::string_view s) { out.append(s); }), cfg);
  // Undo any auto strip_ansi the constructor applied; honor the caller's cfg.
  // (Accomplished by directly re-feeding with a fresh renderer whose sink is
  // purely in-memory — we just need to reset the config override.)
  r.cfg_.strip_ansi = cfg.strip_ansi;
  r.feed(markdown);
  r.flush();
  return out;
}

bool Renderer::is_tty() {
  return stdout_is_tty();
}

} // namespace mdtty
