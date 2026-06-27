"""Tests for evidence interfaces."""

from evicode.evidence import CodePair, SyntaxEvidence
from evicode.fusion import build_report


def test_syntax_evidence_valid_python() -> None:
    """Syntax evidence should score valid Python pairs as valid."""
    pair = CodePair(
        pair_id="x",
        source_language="python",
        target_language="python",
        source_code="def f():\n    return 1\n",
        target_code="def f():\n    return 1\n",
        metadata={},
    )
    result = SyntaxEvidence().run(pair)
    assert result.score == 1.0
    assert result.features["both_syntax_valid"] is True


def test_verification_report_explains_low_evidence() -> None:
    """Verification reports should expose weak evidence as failure modes."""
    pair = CodePair(
        pair_id="bad",
        source_language="python",
        target_language="python",
        source_code="def f():\n    return 1\n",
        target_code="def f(:\n    return 1\n",
        metadata={},
    )
    evidence = [SyntaxEvidence().run(pair)]
    report = build_report(pair.pair_id, 0.1, 0, evidence)
    assert "likely incorrect" in report.summary
    assert "low_syntax_evidence" in report.likely_failure_modes
