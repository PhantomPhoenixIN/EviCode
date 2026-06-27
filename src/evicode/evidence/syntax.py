"""Syntax evidence source."""

from __future__ import annotations

import ast

from evicode.evidence.base import CodePair, EvidenceSource


class SyntaxEvidence(EvidenceSource):
    """Estimate whether source and target code are syntactically valid."""

    name = "syntax"

    def extract(self, pair: CodePair) -> dict[str, float | int | str | bool | None]:
        """Extract syntax-validity features for supported languages."""
        source_valid = self._is_valid_python(pair.source_language, pair.source_code)
        target_valid = self._is_valid_python(pair.target_language, pair.target_code)
        return {
            "source_syntax_valid": source_valid,
            "target_syntax_valid": target_valid,
            "both_syntax_valid": source_valid and target_valid,
        }

    def score(self, features: dict[str, float | int | str | bool | None]) -> float:
        """Score syntax evidence."""
        return 1.0 if bool(features["both_syntax_valid"]) else 0.0

    def explain(self, features: dict[str, float | int | str | bool | None]) -> str:
        """Explain syntax evidence."""
        if features["both_syntax_valid"]:
            return "Both source and target are syntactically valid for supported parsers."
        return "At least one side failed syntax validation or uses an unsupported parser."

    @staticmethod
    def _is_valid_python(language: str, code: str) -> bool:
        """Return Python syntax validity; unsupported languages are marked unknown/valid."""
        if language.lower() != "python":
            return True
        try:
            ast.parse(code)
        except SyntaxError:
            return False
        return True
