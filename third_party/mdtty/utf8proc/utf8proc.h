/*
 * Copyright (c) 2014-2021 Steven G. Johnson, Jiahao Chen, Peter Colberg, Tony Kelman, Scott P. Jones, and other contributors.
 * Copyright (c) 2009 Public Software Group e. V., Berlin, Germany
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef UTF8PROC_H
#define UTF8PROC_H

#define UTF8PROC_VERSION_MAJOR 2

#define UTF8PROC_VERSION_MINOR 9

#define UTF8PROC_VERSION_PATCH 0

#include <stdlib.h>

#if defined(_MSC_VER) && _MSC_VER < 1800

typedef signed char utf8proc_int8_t;
typedef unsigned char utf8proc_uint8_t;
typedef short utf8proc_int16_t;
typedef unsigned short utf8proc_uint16_t;
typedef int utf8proc_int32_t;
typedef unsigned int utf8proc_uint32_t;
#  ifdef _WIN64
typedef __int64 utf8proc_ssize_t;
typedef unsigned __int64 utf8proc_size_t;
#  else
typedef int utf8proc_ssize_t;
typedef unsigned int utf8proc_size_t;
#  endif
#  ifndef __cplusplus

typedef unsigned char utf8proc_bool;
#    ifndef __bool_true_false_are_defined
#      define false 0
#      define true 1
#      define __bool_true_false_are_defined 1
#    endif
#  else
typedef bool utf8proc_bool;
#  endif
#else
#  include <stddef.h>
#  include <stdbool.h>
#  include <inttypes.h>
typedef int8_t utf8proc_int8_t;
typedef uint8_t utf8proc_uint8_t;
typedef int16_t utf8proc_int16_t;
typedef uint16_t utf8proc_uint16_t;
typedef int32_t utf8proc_int32_t;
typedef uint32_t utf8proc_uint32_t;
typedef size_t utf8proc_size_t;
typedef ptrdiff_t utf8proc_ssize_t;
typedef bool utf8proc_bool;
#endif
#include <limits.h>

#ifdef UTF8PROC_STATIC
#  define UTF8PROC_DLLEXPORT
#else
#  ifdef _WIN32
#    ifdef UTF8PROC_EXPORTS
#      define UTF8PROC_DLLEXPORT __declspec(dllexport)
#    else
#      define UTF8PROC_DLLEXPORT __declspec(dllimport)
#    endif
#  elif __GNUC__ >= 4
#    define UTF8PROC_DLLEXPORT __attribute__ ((visibility("default")))
#  else
#    define UTF8PROC_DLLEXPORT
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {

  UTF8PROC_NULLTERM  = (1<<0),

  UTF8PROC_STABLE    = (1<<1),

  UTF8PROC_COMPAT    = (1<<2),

  UTF8PROC_COMPOSE   = (1<<3),

  UTF8PROC_DECOMPOSE = (1<<4),

  UTF8PROC_IGNORE    = (1<<5),

  UTF8PROC_REJECTNA  = (1<<6),

  UTF8PROC_NLF2LS    = (1<<7),

  UTF8PROC_NLF2PS    = (1<<8),

  UTF8PROC_NLF2LF    = (UTF8PROC_NLF2LS | UTF8PROC_NLF2PS),

  UTF8PROC_STRIPCC   = (1<<9),

  UTF8PROC_CASEFOLD  = (1<<10),

  UTF8PROC_CHARBOUND = (1<<11),

  UTF8PROC_LUMP      = (1<<12),

  UTF8PROC_STRIPMARK = (1<<13),

  UTF8PROC_STRIPNA    = (1<<14),
} utf8proc_option_t;

#define UTF8PROC_ERROR_NOMEM -1

#define UTF8PROC_ERROR_OVERFLOW -2

#define UTF8PROC_ERROR_INVALIDUTF8 -3

#define UTF8PROC_ERROR_NOTASSIGNED -4

#define UTF8PROC_ERROR_INVALIDOPTS -5

typedef utf8proc_int16_t utf8proc_propval_t;

typedef struct utf8proc_property_struct {

  utf8proc_propval_t category;
  utf8proc_propval_t combining_class;

  utf8proc_propval_t bidi_class;

  utf8proc_propval_t decomp_type;
  utf8proc_uint16_t decomp_seqindex;
  utf8proc_uint16_t casefold_seqindex;
  utf8proc_uint16_t uppercase_seqindex;
  utf8proc_uint16_t lowercase_seqindex;
  utf8proc_uint16_t titlecase_seqindex;
  utf8proc_uint16_t comb_index;
  unsigned bidi_mirrored:1;
  unsigned comp_exclusion:1;

  unsigned ignorable:1;
  unsigned control_boundary:1;

  unsigned charwidth:2;
  unsigned pad:2;

  unsigned boundclass:6;
  unsigned indic_conjunct_break:2;
} utf8proc_property_t;

typedef enum {
  UTF8PROC_CATEGORY_CN  = 0,
  UTF8PROC_CATEGORY_LU  = 1,
  UTF8PROC_CATEGORY_LL  = 2,
  UTF8PROC_CATEGORY_LT  = 3,
  UTF8PROC_CATEGORY_LM  = 4,
  UTF8PROC_CATEGORY_LO  = 5,
  UTF8PROC_CATEGORY_MN  = 6,
  UTF8PROC_CATEGORY_MC  = 7,
  UTF8PROC_CATEGORY_ME  = 8,
  UTF8PROC_CATEGORY_ND  = 9,
  UTF8PROC_CATEGORY_NL = 10,
  UTF8PROC_CATEGORY_NO = 11,
  UTF8PROC_CATEGORY_PC = 12,
  UTF8PROC_CATEGORY_PD = 13,
  UTF8PROC_CATEGORY_PS = 14,
  UTF8PROC_CATEGORY_PE = 15,
  UTF8PROC_CATEGORY_PI = 16,
  UTF8PROC_CATEGORY_PF = 17,
  UTF8PROC_CATEGORY_PO = 18,
  UTF8PROC_CATEGORY_SM = 19,
  UTF8PROC_CATEGORY_SC = 20,
  UTF8PROC_CATEGORY_SK = 21,
  UTF8PROC_CATEGORY_SO = 22,
  UTF8PROC_CATEGORY_ZS = 23,
  UTF8PROC_CATEGORY_ZL = 24,
  UTF8PROC_CATEGORY_ZP = 25,
  UTF8PROC_CATEGORY_CC = 26,
  UTF8PROC_CATEGORY_CF = 27,
  UTF8PROC_CATEGORY_CS = 28,
  UTF8PROC_CATEGORY_CO = 29,
} utf8proc_category_t;

typedef enum {
  UTF8PROC_BIDI_CLASS_L     = 1,
  UTF8PROC_BIDI_CLASS_LRE   = 2,
  UTF8PROC_BIDI_CLASS_LRO   = 3,
  UTF8PROC_BIDI_CLASS_R     = 4,
  UTF8PROC_BIDI_CLASS_AL    = 5,
  UTF8PROC_BIDI_CLASS_RLE   = 6,
  UTF8PROC_BIDI_CLASS_RLO   = 7,
  UTF8PROC_BIDI_CLASS_PDF   = 8,
  UTF8PROC_BIDI_CLASS_EN    = 9,
  UTF8PROC_BIDI_CLASS_ES   = 10,
  UTF8PROC_BIDI_CLASS_ET   = 11,
  UTF8PROC_BIDI_CLASS_AN   = 12,
  UTF8PROC_BIDI_CLASS_CS   = 13,
  UTF8PROC_BIDI_CLASS_NSM  = 14,
  UTF8PROC_BIDI_CLASS_BN   = 15,
  UTF8PROC_BIDI_CLASS_B    = 16,
  UTF8PROC_BIDI_CLASS_S    = 17,
  UTF8PROC_BIDI_CLASS_WS   = 18,
  UTF8PROC_BIDI_CLASS_ON   = 19,
  UTF8PROC_BIDI_CLASS_LRI  = 20,
  UTF8PROC_BIDI_CLASS_RLI  = 21,
  UTF8PROC_BIDI_CLASS_FSI  = 22,
  UTF8PROC_BIDI_CLASS_PDI  = 23,
} utf8proc_bidi_class_t;

typedef enum {
  UTF8PROC_DECOMP_TYPE_FONT      = 1,
  UTF8PROC_DECOMP_TYPE_NOBREAK   = 2,
  UTF8PROC_DECOMP_TYPE_INITIAL   = 3,
  UTF8PROC_DECOMP_TYPE_MEDIAL    = 4,
  UTF8PROC_DECOMP_TYPE_FINAL     = 5,
  UTF8PROC_DECOMP_TYPE_ISOLATED  = 6,
  UTF8PROC_DECOMP_TYPE_CIRCLE    = 7,
  UTF8PROC_DECOMP_TYPE_SUPER     = 8,
  UTF8PROC_DECOMP_TYPE_SUB       = 9,
  UTF8PROC_DECOMP_TYPE_VERTICAL = 10,
  UTF8PROC_DECOMP_TYPE_WIDE     = 11,
  UTF8PROC_DECOMP_TYPE_NARROW   = 12,
  UTF8PROC_DECOMP_TYPE_SMALL    = 13,
  UTF8PROC_DECOMP_TYPE_SQUARE   = 14,
  UTF8PROC_DECOMP_TYPE_FRACTION = 15,
  UTF8PROC_DECOMP_TYPE_COMPAT   = 16,
} utf8proc_decomp_type_t;

typedef enum {
  UTF8PROC_BOUNDCLASS_START              =  0,
  UTF8PROC_BOUNDCLASS_OTHER              =  1,
  UTF8PROC_BOUNDCLASS_CR                 =  2,
  UTF8PROC_BOUNDCLASS_LF                 =  3,
  UTF8PROC_BOUNDCLASS_CONTROL            =  4,
  UTF8PROC_BOUNDCLASS_EXTEND             =  5,
  UTF8PROC_BOUNDCLASS_L                  =  6,
  UTF8PROC_BOUNDCLASS_V                  =  7,
  UTF8PROC_BOUNDCLASS_T                  =  8,
  UTF8PROC_BOUNDCLASS_LV                 =  9,
  UTF8PROC_BOUNDCLASS_LVT                = 10,
  UTF8PROC_BOUNDCLASS_REGIONAL_INDICATOR = 11,
  UTF8PROC_BOUNDCLASS_SPACINGMARK        = 12,
  UTF8PROC_BOUNDCLASS_PREPEND            = 13,
  UTF8PROC_BOUNDCLASS_ZWJ                = 14,

  UTF8PROC_BOUNDCLASS_E_BASE             = 15,
  UTF8PROC_BOUNDCLASS_E_MODIFIER         = 16,
  UTF8PROC_BOUNDCLASS_GLUE_AFTER_ZWJ     = 17,
  UTF8PROC_BOUNDCLASS_E_BASE_GAZ         = 18,

  UTF8PROC_BOUNDCLASS_EXTENDED_PICTOGRAPHIC = 19,
  UTF8PROC_BOUNDCLASS_E_ZWG = 20,
} utf8proc_boundclass_t;

typedef enum {
  UTF8PROC_INDIC_CONJUNCT_BREAK_NONE = 0,
  UTF8PROC_INDIC_CONJUNCT_BREAK_LINKER = 1,
  UTF8PROC_INDIC_CONJUNCT_BREAK_CONSONANT = 2,
  UTF8PROC_INDIC_CONJUNCT_BREAK_EXTEND = 3,
} utf8proc_indic_conjunct_break_t;

typedef utf8proc_int32_t (*utf8proc_custom_func)(utf8proc_int32_t codepoint, void *data);

UTF8PROC_DLLEXPORT extern const utf8proc_int8_t utf8proc_utf8class[256];

UTF8PROC_DLLEXPORT const char *utf8proc_version(void);

UTF8PROC_DLLEXPORT const char *utf8proc_unicode_version(void);

UTF8PROC_DLLEXPORT const char *utf8proc_errmsg(utf8proc_ssize_t errcode);

UTF8PROC_DLLEXPORT utf8proc_ssize_t utf8proc_iterate(const utf8proc_uint8_t *str, utf8proc_ssize_t strlen, utf8proc_int32_t *codepoint_ref);

UTF8PROC_DLLEXPORT utf8proc_bool utf8proc_codepoint_valid(utf8proc_int32_t codepoint);

UTF8PROC_DLLEXPORT utf8proc_ssize_t utf8proc_encode_char(utf8proc_int32_t codepoint, utf8proc_uint8_t *dst);

UTF8PROC_DLLEXPORT const utf8proc_property_t *utf8proc_get_property(utf8proc_int32_t codepoint);

UTF8PROC_DLLEXPORT utf8proc_ssize_t utf8proc_decompose_char(
  utf8proc_int32_t codepoint, utf8proc_int32_t *dst, utf8proc_ssize_t bufsize,
  utf8proc_option_t options, int *last_boundclass
);

UTF8PROC_DLLEXPORT utf8proc_ssize_t utf8proc_decompose(
  const utf8proc_uint8_t *str, utf8proc_ssize_t strlen,
  utf8proc_int32_t *buffer, utf8proc_ssize_t bufsize, utf8proc_option_t options
);

UTF8PROC_DLLEXPORT utf8proc_ssize_t utf8proc_decompose_custom(
  const utf8proc_uint8_t *str, utf8proc_ssize_t strlen,
  utf8proc_int32_t *buffer, utf8proc_ssize_t bufsize, utf8proc_option_t options,
  utf8proc_custom_func custom_func, void *custom_data
);

UTF8PROC_DLLEXPORT utf8proc_ssize_t utf8proc_normalize_utf32(utf8proc_int32_t *buffer, utf8proc_ssize_t length, utf8proc_option_t options);

UTF8PROC_DLLEXPORT utf8proc_ssize_t utf8proc_reencode(utf8proc_int32_t *buffer, utf8proc_ssize_t length, utf8proc_option_t options);

UTF8PROC_DLLEXPORT utf8proc_bool utf8proc_grapheme_break_stateful(
    utf8proc_int32_t codepoint1, utf8proc_int32_t codepoint2, utf8proc_int32_t *state);

UTF8PROC_DLLEXPORT utf8proc_bool utf8proc_grapheme_break(
    utf8proc_int32_t codepoint1, utf8proc_int32_t codepoint2);

UTF8PROC_DLLEXPORT utf8proc_int32_t utf8proc_tolower(utf8proc_int32_t c);

UTF8PROC_DLLEXPORT utf8proc_int32_t utf8proc_toupper(utf8proc_int32_t c);

UTF8PROC_DLLEXPORT utf8proc_int32_t utf8proc_totitle(utf8proc_int32_t c);

UTF8PROC_DLLEXPORT int utf8proc_islower(utf8proc_int32_t c);

UTF8PROC_DLLEXPORT int utf8proc_isupper(utf8proc_int32_t c);

UTF8PROC_DLLEXPORT int utf8proc_charwidth(utf8proc_int32_t codepoint);

UTF8PROC_DLLEXPORT utf8proc_category_t utf8proc_category(utf8proc_int32_t codepoint);

UTF8PROC_DLLEXPORT const char *utf8proc_category_string(utf8proc_int32_t codepoint);

UTF8PROC_DLLEXPORT utf8proc_ssize_t utf8proc_map(
  const utf8proc_uint8_t *str, utf8proc_ssize_t strlen, utf8proc_uint8_t **dstptr, utf8proc_option_t options
);

UTF8PROC_DLLEXPORT utf8proc_ssize_t utf8proc_map_custom(
  const utf8proc_uint8_t *str, utf8proc_ssize_t strlen, utf8proc_uint8_t **dstptr, utf8proc_option_t options,
  utf8proc_custom_func custom_func, void *custom_data
);

UTF8PROC_DLLEXPORT utf8proc_uint8_t *utf8proc_NFD(const utf8proc_uint8_t *str);

UTF8PROC_DLLEXPORT utf8proc_uint8_t *utf8proc_NFC(const utf8proc_uint8_t *str);

UTF8PROC_DLLEXPORT utf8proc_uint8_t *utf8proc_NFKD(const utf8proc_uint8_t *str);

UTF8PROC_DLLEXPORT utf8proc_uint8_t *utf8proc_NFKC(const utf8proc_uint8_t *str);

UTF8PROC_DLLEXPORT utf8proc_uint8_t *utf8proc_NFKC_Casefold(const utf8proc_uint8_t *str);

#ifdef __cplusplus
}
#endif

#endif
