"""Build prompt context from a folder of local files.

No network calls and no embedding model: relevance ranking is done with a
simple keyword-overlap score so the whole pipeline stays usable fully offline
with nothing beyond the chat model itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Directories that are never worth scanning: VCS metadata, dependency
# caches, build output, virtualenvs.
IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "models",
}

# Extensions treated as text and worth indexing. Anything else is skipped
# rather than risking a binary file being read as text.
TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".json",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".sh", ".bash", ".zsh",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".java", ".kt", ".go", ".rs",
    ".rb", ".php", ".sql", ".html", ".css", ".xml", ".env", ".gitignore",
    ".dockerfile", ".makefile", ".csv", ".log",
}

MAX_FILE_BYTES = 512_000  # skip anything larger than ~500KB
CHUNK_CHARS = 1_500
CHUNK_OVERLAP = 200
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class Chunk:
    path: Path
    start_line: int
    text: str

    def render(self) -> str:
        return f"### {self.path} (from line {self.start_line})\n{self.text.strip()}\n"


def _is_probably_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    if path.name.lower() in {"dockerfile", "makefile", "license", "readme"}:
        return True
    return False


def iter_files(folder: Path):
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIR_NAMES for part in path.parts):
            continue
        if not _is_probably_text(path):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None


def chunk_file(path: Path, text: str):
    lines = text.splitlines()
    if not lines:
        return
    joined = "\n".join(lines)
    pos = 0
    line_starts = []
    running = 0
    for line in lines:
        line_starts.append(running)
        running += len(line) + 1

    while pos < len(joined):
        end = min(pos + CHUNK_CHARS, len(joined))
        chunk_text = joined[pos:end]
        start_line = 1
        for idx, offset in enumerate(line_starts):
            if offset > pos:
                break
            start_line = idx + 1
        yield Chunk(path=path, start_line=start_line, text=chunk_text)
        if end == len(joined):
            break
        pos = end - CHUNK_OVERLAP


def collect_chunks(folder: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in iter_files(folder):
        text = _read_text(path)
        if not text or not text.strip():
            continue
        chunks.extend(chunk_file(path, text))
    return chunks


def _score(chunk_text: str, query_words: set[str]) -> int:
    if not query_words:
        return 0
    chunk_words = WORD_RE.findall(chunk_text.lower())
    return sum(1 for w in chunk_words if w in query_words)


def build_context(folder: Path, prompt: str, max_chars: int) -> str:
    """Pick the chunks most relevant to `prompt`, up to `max_chars` total."""
    chunks = collect_chunks(folder)
    if not chunks:
        return ""

    query_words = {w for w in WORD_RE.findall(prompt.lower()) if len(w) > 2}
    scored = [(_score(c.text, query_words), i, c) for i, c in enumerate(chunks)]
    # Keep original file order as a tiebreak so unrelated-query runs still
    # produce a stable, readable context instead of arbitrary ordering.
    scored.sort(key=lambda t: (-t[0], t[1]))

    selected: list[Chunk] = []
    total = 0
    for score, _, chunk in scored:
        rendered = chunk.render()
        if total + len(rendered) > max_chars and selected:
            break
        selected.append(chunk)
        total += len(rendered)
        if total >= max_chars:
            break

    # Restore file order for readability in the final prompt.
    selected.sort(key=lambda c: (str(c.path), c.start_line))
    return "\n".join(c.render() for c in selected)
