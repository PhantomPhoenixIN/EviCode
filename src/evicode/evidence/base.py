"""Base abstractions for evidence extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvidenceResult:
    """Structured output emitted by an evidence source."""

    name: str
    features: dict[str, float | int | str | bool | None]
    score: float
    explanation: str


@dataclass(frozen=True)
class CodePair:
    """Source-target code pair used by evidence modules."""

    pair_id: str
    source_language: str
    target_language: str
    source_code: str
    target_code: str
    metadata: dict[str, Any]


class EvidenceSource(ABC):
    """Interface implemented by every EviCode evidence source."""

    name: str

    @abstractmethod
    def extract(self, pair: CodePair) -> dict[str, float | int | str | bool | None]:
        """Extract raw evidence features from a code pair."""

    @abstractmethod
    def score(self, features: dict[str, float | int | str | bool | None]) -> float:
        """Convert extracted features into a normalized score in [0, 1]."""

    @abstractmethod
    def explain(self, features: dict[str, float | int | str | bool | None]) -> str:
        """Return a concise human-readable explanation for the evidence."""

    def run(self, pair: CodePair) -> EvidenceResult:
        """Extract, score, and explain evidence for one code pair."""
        features = self.extract(pair)
        return EvidenceResult(
            name=self.name,
            features=features,
            score=self.score(features),
            explanation=self.explain(features),
        )
