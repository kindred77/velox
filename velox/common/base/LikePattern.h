// SQL LIKE conversion extracted from Velox Filter.h.
// Copyright (c) Facebook, Inc. and its affiliates.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace facebook::velox::common::sql {

/// Converts a SQL LIKE pattern to an anchored RE2 regular expression,
/// shared by optimizer safety proofs, scan filters and runtime LIKE.
/// `%` matches any
/// sequence, `_` matches a single character; `escapeChar` (when present)
/// escapes `%`, `_` and itself. Sets `validPattern` to false for a trailing
/// escape or an escaped character that is not `%`, `_` or the escape
/// character.
inline std::string likePatternToRe2(
    std::string_view pattern,
    std::optional<char> escapeChar,
    bool& validPattern) {
  std::string regex;
  validPattern = true;
  regex.reserve(pattern.size() * 2);
  regex.append("^");
  bool escaped = false;
  for (const char c : pattern) {
    if (escaped && !(c == '%' || c == '_' || c == escapeChar)) {
      validPattern = false;
    }
    if (!escaped && c == escapeChar) {
      escaped = true;
    } else {
      switch (c) {
        case '%':
          regex.append(escaped ? "%" : ".*");
          escaped = false;
          break;
        case '_':
          regex.append(escaped ? "_" : ".");
          escaped = false;
          break;
        // Escape all the meta characters in re2.
        case '\\':
        case '|':
        case '^':
        case '$':
        case '.':
        case '*':
        case '+':
        case '?':
        case '(':
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
          regex.append("\\"); // Append the meta character after the escape.
          [[fallthrough]];
        default:
          regex.append(1, c);
          escaped = false;
      }
    }
  }
  if (escaped) {
    validPattern = false;
  }

  regex.append("$");
  return regex;
}

} // namespace facebook::velox::common::sql
