"""Evidence-grounded verification reports."""

from __future__ import annotations

from dataclasses import dataclass

from evicode.evidence import EvidenceResult


@dataclass(frozen=True)
class VerificationReport:
    """Explainable verification output for a code pair."""

    pair_id: str
    score: float
    prediction: int
    evidence: list[EvidenceResult]
    summary: str
    likely_failure_modes: list[str]


def build_report(
    pair_id: str,
    score: float,
    prediction: int,
    evidence: list[EvidenceResult],
    threshold: float = 0.5,
) -> VerificationReport:
    """Build an evidence-grounded report from module outputs."""
    low = [item.name for item in evidence if item.score < threshold]
    high = [item.name for item in evidence if item.score >= threshold]
    failure_modes = [f"low_{name}_evidence" for name in low]
    if prediction:
        verdict = "likely correct"
    else:
        verdict = "likely incorrect"
    summary = (
        f"Overall verdict: {verdict} (score={score:.3f}). "
        f"High-confidence evidence: {', '.join(high) if high else 'none'}. "
        f"Weak evidence: {', '.join(low) if low else 'none'}."
    )
    return VerificationReport(
        pair_id=pair_id,
        score=score,
        prediction=prediction,
        evidence=evidence,
        summary=summary,
        likely_failure_modes=failure_modes,
    )
