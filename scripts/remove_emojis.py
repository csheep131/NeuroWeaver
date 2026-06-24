#!/usr/bin/env python3
"""Remove all emoji characters from Markdown files in the project.

Preserves all text content, structure, and formatting — only strips
emoji Unicode code points and cleans up residual double spaces.

Important: \\p{Emoji} includes ASCII digits 0-9, '#' and '*' because
they participate in keycap sequences (1 etc.). We explicitly exclude
those ASCII characters so only true emoji glyphs are removed.
"""

import re
import regex
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Match emoji characters but EXCLUDE ASCII digits, '#' and '*'
# which are technically part of \p{Emoji} due to keycap sequences.
# Also exclude U+FE0E (text-style selector) from removal since it
# only affects presentation of non-emoji characters.
#
# Breakdown:
# \p{Emoji_Presentation} — characters that render as emoji by default
# \p{Emoji_Modifier} — skin-tone modifiers (U+1F3FB..1F3FF)
# \uFE0F — Variation Selector-16 (emoji presentation)
# \u200D — Zero Width Joiner (compound emoji sequences)
# \u20E3 — Combining Enclosing Keycap
# [\p{Emoji}--\p{ASCII}] — non-ASCII emoji code points
# (symbols like , , arrows, etc.)
EMOJI_PATTERN = regex.compile(
r"\p{Emoji_Presentation}"
r"|\p{Emoji_Modifier}"
r"|\uFE0F"
r"|\u200D"
r"|\u20E3"
r"|(?:[\p{Emoji}--\p{ASCII}])"
)

# Secondary cleanup: collapse runs of 2+ spaces into one
MULTI_SPACE_PATTERN = re.compile(r" +")


def strip_emojis(text: str) -> str:
"""Remove all emoji characters from text, clean up spacing."""
cleaned = EMOJI_PATTERN.sub("", text)
# Collapse multiple spaces but preserve leading indentation
lines = cleaned.split("\n")
result_lines = []
for line in lines:
stripped = line.lstrip(" ")
leading = line[: len(line) - len(stripped)]
cleaned_line = leading + MULTI_SPACE_PATTERN.sub(" ", stripped).strip()
result_lines.append(cleaned_line)
return "\n".join(result_lines)


def count_emojis(text: str) -> int:
"""Count emoji characters in text."""
return len(EMOJI_PATTERN.findall(text))


def main() -> None:
md_files = sorted(PROJECT_ROOT.rglob("*.md"))

# Skip files inside .venv, node_modules, .git
md_files = [
f for f in md_files
if not any(
part.startswith((".", "node_modules"))
for part in f.relative_to(PROJECT_ROOT).parts
)
]

total_emojis_before = 0
total_files_changed = 0
total_emojis_after = 0

for md_file in md_files:
try:
content = md_file.read_text(encoding="utf-8")
except (UnicodeDecodeError, PermissionError) as exc:
print(f" SKIP {md_file.relative_to(PROJECT_ROOT)}: {exc}")
continue

emoji_count = count_emojis(content)
total_emojis_before += emoji_count

if emoji_count == 0:
continue

cleaned = strip_emojis(content)
remaining = count_emojis(cleaned)
total_emojis_after += remaining

md_file.write_text(cleaned, encoding="utf-8")
total_files_changed += 1

rel = md_file.relative_to(PROJECT_ROOT)
print(f" CLEANED {rel}: removed {emoji_count} emojis ({remaining} remaining)")

print(f"\n{'='*60}")
print(f"Total files scanned: {len(md_files)}")
print(f"Total files changed: {total_files_changed}")
print(f"Total emojis before: {total_emojis_before}")
print(f"Total emojis remaining: {total_emojis_after}")
print(f"{'='*60}")


if __name__ == "__main__":
main()
