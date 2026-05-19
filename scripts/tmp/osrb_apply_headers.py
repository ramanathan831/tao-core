#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-shot OSRB header normalizer for tao-core Apache-2.0 .py files.

Replaces the legacy verbose Apache boilerplate (Copyright NVIDIA CORPORATION.
+ full License preamble) with the Confluence-mandated 2-line SPDX header.
Adds the SPDX header to files that lack any header. Preserves shebangs,
encoding declarations, and "Original source taken from..." attribution lines.

Skips telemetry_gateway/ (proprietary).
"""
import os
import re
import sys

SKIP_DIRS = {".git", "telemetry_gateway"}

SPDX_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) {year} NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)

COPY_RE = re.compile(
    r"^# Copyright \(c\) (?P<year>\d{4})[, ]?\s*NVIDIA CORPORATION\.?\s+All rights reserved\.",
    re.IGNORECASE,
)
LICENSE_TAIL_RE = re.compile(r"^# limitations under the License\.\s*$")
ORIG_RE = re.compile(r"^# Original source taken from .*$")
ENCODING_RE = re.compile(r"coding[:=]\s*([-\w.]+)")


def split_prelude(lines):
    """Return (prelude_lines, rest_lines). prelude = shebang + optional encoding."""
    i = 0
    if i < len(lines) and lines[i].startswith("#!"):
        i += 1
    # optional encoding declaration (must be in first 2 lines per PEP 263)
    if i < len(lines) and ENCODING_RE.search(lines[i]):
        i += 1
    return lines[:i], lines[i:]


def find_old_header_block(lines):
    """Locate the legacy NVIDIA Apache block within `lines`.

    Returns (start, end, original_attribution_lines) or None.
    start..end is the half-open slice to remove.
    """
    n = len(lines)
    i = 0
    # skip blank lines at front of rest
    while i < n and lines[i].strip() == "":
        i += 1
    if i >= n:
        return None
    m = COPY_RE.match(lines[i].rstrip("\n"))
    if not m:
        return None
    start = i
    year = m.group("year")
    # extract any "Original source taken from" lines within the block
    attribution = []
    j = i + 1
    end = None
    while j < n:
        line = lines[j].rstrip("\n")
        if ORIG_RE.match(line):
            attribution.append(line)
        if LICENSE_TAIL_RE.match(line):
            end = j + 1
            break
        j += 1
        if j - i > 30:  # safety guard
            return None
    if end is None:
        return None
    return start, end, attribution, year


def has_existing_spdx(lines):
    head = "".join(lines[:6])
    return "SPDX-FileCopyrightText" in head and "NVIDIA CORPORATION & AFFILIATES" in head


def process_file(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines(keepends=True)
    prelude, rest = split_prelude(lines)

    if has_existing_spdx(rest):
        return "skip_already_spdx"

    blk = find_old_header_block(rest)
    if blk is not None:
        start, end, attribution, year = blk
        spdx = SPDX_HEADER.format(year=year)
        new_block_lines = [spdx]
        if attribution:
            new_block_lines.append("#\n")
            for a in attribution:
                new_block_lines.append(a + "\n")
        # preserve any blank lines before start
        pre_blanks = rest[:start]
        post = rest[end:]
        new_rest = pre_blanks + new_block_lines + post
        new_text = "".join(prelude) + "".join(new_rest)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new_text)
        return "replaced"

    # no_header: prepend SPDX (use 2024)
    spdx = SPDX_HEADER.format(year="2024")
    # if rest starts with content (e.g. docstring) preserve blank separator
    sep = "\n" if rest and rest[0].strip() != "" else ""
    new_text = "".join(prelude) + spdx + sep + "".join(rest)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_text)
    return "prepended"


def main():
    root = "."
    counts = {"replaced": 0, "prepended": 0, "skip_already_spdx": 0, "untouched": 0}
    issues = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        # also skip any path containing telemetry_gateway
        if "telemetry_gateway" in dirpath.split(os.sep):
            continue
        for f in filenames:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dirpath, f)
            try:
                result = process_file(p)
            except Exception as e:
                issues.append((p, str(e)))
                continue
            counts[result] = counts.get(result, 0) + 1
    print("Counts:", counts)
    if issues:
        print("Issues:")
        for p, e in issues:
            print(" ", p, "->", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
