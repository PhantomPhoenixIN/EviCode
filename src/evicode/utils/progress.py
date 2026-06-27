"""Resume-safe progress helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_progress_dirs(root: Path) -> None:
    """Create progress directories if they do not exist."""
    (root / "progress" / "logs").mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    """Read JSON from path or return a default value."""
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    """Write JSON atomically enough for small status files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def mark_stage(output_dir: Path, stage: str, status: str, details: dict[str, Any]) -> None:
    """Record a stage status file."""
    write_json(
        output_dir / "status" / f"{stage}.json",
        {"stage": stage, "status": status, "details": details},
    )
