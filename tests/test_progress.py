"""Tests for progress helpers."""

from evicode.utils.progress import read_json, write_json


def test_write_and_read_json(tmp_path) -> None:
    """Progress JSON helpers should round-trip data."""
    path = tmp_path / "progress.json"
    write_json(path, {"ok": True})
    assert read_json(path, {}) == {"ok": True}
