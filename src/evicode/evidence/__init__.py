"""Evidence source registry."""

from evicode.evidence.base import CodePair, EvidenceResult, EvidenceSource
from evicode.evidence.simple import SimilarityEvidence
from evicode.evidence.syntax import SyntaxEvidence


def build_evidence_sources(names: list[str]) -> list[EvidenceSource]:
    """Build evidence sources by name."""
    sources: list[EvidenceSource] = []
    for name in names:
        if name == "syntax":
            sources.append(SyntaxEvidence())
        else:
            sources.append(SimilarityEvidence(name=name))
    return sources


__all__ = [
    "CodePair",
    "EvidenceResult",
    "EvidenceSource",
    "SimilarityEvidence",
    "SyntaxEvidence",
    "build_evidence_sources",
]
