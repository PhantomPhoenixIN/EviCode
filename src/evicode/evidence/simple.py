"""Lightweight placeholder evidence sources for the initial smoke pipeline."""

from __future__ import annotations

from difflib import SequenceMatcher

from evicode.evidence.base import CodePair, EvidenceSource


class SimilarityEvidence(EvidenceSource):
    """Token/text similarity evidence used until richer modules are implemented."""

    def __init__(self, name: str) -> None:
        """Create a named similarity evidence source."""
        self.name = name

    def extract(self, pair: CodePair) -> dict[str, float | int | str | bool | None]:
        """Extract a simple normalized string similarity feature."""
        ratio = SequenceMatcher(None, pair.source_code, pair.target_code).ratio()
        return {"similarity": ratio}

    def score(self, features: dict[str, float | int | str | bool | None]) -> float:
        """Return the similarity feature as a score."""
        return float(features["similarity"])

    def explain(self, features: dict[str, float | int | str | bool | None]) -> str:
        """Explain the similarity score."""
        return f"{self.name} proxy similarity={float(features['similarity']):.3f}."
