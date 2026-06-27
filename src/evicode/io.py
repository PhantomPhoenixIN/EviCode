"""JSONL IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
