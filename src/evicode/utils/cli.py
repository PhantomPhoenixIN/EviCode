"""Shared command-line helpers."""

from __future__ import annotations

import argparse


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the required resume-safe CLI arguments to a parser."""
    parser.add_argument("--resume", action="store_true", help="Resume and skip valid outputs.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing outputs.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    return parser
