"""Offline ingestion step 1: turn raw curriculum markdown into chunks.

Source = Notion (exported or pulled) -> data/curriculum/**/*.md
Use header-aware splitting so a lesson's section stays intact, then size-cap.

Why a hand-rolled scanner instead of MarkdownHeaderTextSplitter:
  * The Notion export overloads the `#` level. Day dividers
    ("Saturday August 17, 2024 - (Week 1- Day 1)") AND topic headings
    ("Understanding Computer Hardware") are BOTH level-1, so a splitter that
    keys metadata off heading level can't tell the module from the lesson.
  * Many `#` lines are really code comments inside ``` ``` fences
    (`# main.py`, `# Create an S3 client`). They must NOT be read as headings.
A code-fence-aware line scanner handles both correctly and is trivial to test.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Target ~800 chars/chunk (~200 tokens) with overlap. Short sections stay whole;
# only sections that exceed the cap get size-split. See docs/BUILD-PLAN.md.
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "120"))

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
# Some shell output in the notes is unfenced and prefixed with `#`, so it parses
# as a heading by spec (e.g. `# docker:x:994:ec2-user`). A host:port / uid:gid
# colon-digit signature never appears in a real lesson title, so demote it.
_NOT_A_HEADING = re.compile(r":\d|::")
# A level-1 heading that divides the curriculum into modules: a dated day
# ("... (Week 1- Day 1)") or one of the end-of-course project sections.
_DAY = re.compile(r"week\s*\d+\s*-?\s*day\s*\d+", re.IGNORECASE)
_PROJECT = re.compile(r"^AWS Cloud Engineer (Projects|Tech Challenge)", re.IGNORECASE)


def _clean(text: str) -> str:
    """Strip Notion markdown bold and stray hashes from a heading label."""
    return re.sub(r"\s+", " ", text.replace("**", "").strip("# ").strip())


def _is_module_divider(level: int, text: str) -> bool:
    return level == 1 and (bool(_DAY.search(text)) or bool(_PROJECT.match(text)))


def _has_body(lines: list[str]) -> bool:
    """True if any line after the heading carries real content (not just `---`)."""
    return any(re.search(r"[A-Za-z0-9]", line) for line in lines[1:])


def _split_sections(markdown_text: str):
    """Yield (module, lesson, content) for each heading-delimited section.

    Tracks fenced-code state so `#` comments inside ``` blocks are treated as
    content, never headings. `module` follows the most recent day/project
    divider.

    Notion flattens parent topics and their sub-points to the same heading level,
    so a topic heading is often immediately followed by another heading with no
    body of its own. Rather than emit those as tiny standalone chunks, heading-only
    sections are carried forward and merged into the next content-bearing section;
    the outermost carried heading becomes that chunk's `lesson` (it is the better
    citation label), and every carried heading is prepended to the content.
    """
    module: str | None = None
    cur_level = cur_label = None
    body: list[str] = []
    pending: list[str] = []  # headings of heading-only sections awaiting a body
    in_code = False

    def flush():
        if cur_label is None:
            return None
        is_divider = _is_module_divider(cur_level, cur_label)
        if not _has_body(body):
            # Heading-only: carry non-divider headings into the next section.
            # Dividers carry no text forward; they have already set `module`.
            if not is_divider:
                pending.append(cur_label)
            return None
        if pending:
            lesson = _clean(pending[0])
            content = "\n".join([_clean(h) for h in pending] + body).strip()
            pending.clear()
        else:
            lesson = "Overview" if is_divider else _clean(cur_label)
            content = "\n".join(body).strip()
        return (module or "Course Overview", lesson, content) if content else None

    for line in markdown_text.splitlines():
        if _FENCE.match(line):
            in_code = not in_code
            body.append(line)
            continue

        m = None if in_code else _HEADING.match(line)
        if not m or _NOT_A_HEADING.search(m.group(2)):
            body.append(line)
            continue

        # Real heading: close the previous section before opening this one.
        section = flush()
        if section:
            yield section

        level, raw = len(m.group(1)), m.group(2)
        if _is_module_divider(level, raw):
            module = _clean(raw)
        cur_level, cur_label = level, raw
        body = [line]  # keep the heading in the chunk so it carries topic words

    section = flush()
    if section:
        yield section


def chunk_markdown(path: str, source_url: str | None = None) -> list[dict]:
    """Return [{content, module, lesson, source_url}, ...] for one markdown file.

    Sections stay whole unless they exceed CHUNK_SIZE, in which case they are
    size-split with overlap (metadata is copied onto every resulting piece).
    """
    text = Path(path).read_text(encoding="utf-8")
    source_url = source_url or os.environ.get("CURRICULUM_SOURCE_URL") or Path(path).name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=False
    )

    chunks: list[dict] = []
    for module, lesson, content in _split_sections(text):
        pieces = splitter.split_text(content) if len(content) > CHUNK_SIZE else [content]
        # Prepend a `module > lesson` breadcrumb so each chunk's embedding carries
        # its place in the curriculum, not just the local prose. (Lightweight
        # contextual retrieval — the breadcrumb is also the citation.)
        breadcrumb = f"{module} > {lesson}"
        for piece in pieces:
            piece = piece.strip()
            if not re.search(r"[A-Za-z0-9]", piece):
                continue  # drop divider-only fragments left by size-splitting
            chunks.append(
                {
                    "content": f"{breadcrumb}\n\n{piece}",
                    "module": module,
                    "lesson": lesson,
                    "source_url": source_url,
                }
            )
    return chunks


def find_curriculum_files(root: str = "data/curriculum") -> list[str]:
    """Recursively find curriculum markdown (Notion export nests it in subdirs)."""
    return sorted(glob.glob(os.path.join(root, "**", "*.md"), recursive=True))


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] or find_curriculum_files()
    total = 0
    for p in paths:
        cs = chunk_markdown(p)
        total += len(cs)
        mods = sorted({c["module"] for c in cs})
        print(f"{p}\n  -> {len(cs)} chunks across {len(mods)} modules")
    print(f"TOTAL: {total} chunks")
