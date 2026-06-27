"""Evidence taxonomy used by experiments, analysis, and paper artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvidenceSource:
    """Metadata for one evidence source."""

    name: str
    feature: str
    category: str
    output_type: str
    cost_level: str
    description: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-serializable metadata."""
        return asdict(self)


EVIDENCE_SOURCES = [
    EvidenceSource(
        "Token overlap",
        "token_jaccard",
        "Lexical",
        "continuous",
        "low",
        "Jaccard overlap over language-agnostic lexical tokens.",
    ),
    EvidenceSource(
        "Edit similarity",
        "edit_similarity",
        "Lexical",
        "continuous",
        "low",
        "Normalized sequence similarity over source and target code text.",
    ),
    EvidenceSource(
        "Length ratio",
        "length_ratio",
        "Lexical",
        "continuous",
        "low",
        "Ratio between shorter and longer program length.",
    ),
    EvidenceSource(
        "Parse validity",
        "syntax_proxy",
        "Syntactic",
        "binary",
        "low",
        "Tree-sitter parse validity for both source and target programs.",
    ),
    EvidenceSource(
        "AST node similarity",
        "ast_similarity",
        "Syntactic",
        "continuous",
        "medium",
        "Cosine similarity over parser node-type frequency vectors.",
    ),
    EvidenceSource(
        "AST depth similarity",
        "ast_depth_similarity",
        "Syntactic",
        "continuous",
        "medium",
        "Similarity of maximum parse-tree depth; an approximation, not tree edit distance.",
    ),
    EvidenceSource(
        "AST shape similarity",
        "ast_shape_similarity",
        "Syntactic",
        "continuous",
        "medium",
        "Similarity of depth histograms from parse trees.",
    ),
    EvidenceSource(
        "Control flow",
        "control_flow_similarity",
        "Structural",
        "continuous",
        "low",
        "Similarity over branch, loop, return, and exception keyword counts.",
    ),
    EvidenceSource(
        "Nesting depth",
        "nesting_depth_similarity",
        "Structural",
        "continuous",
        "low",
        "Similarity of approximate brace/indentation nesting depth.",
    ),
    EvidenceSource(
        "Condition/operator patterns",
        "operator_pattern_similarity",
        "Structural",
        "continuous",
        "low",
        "Similarity of conditional and boolean/comparison operator patterns.",
    ),
    EvidenceSource(
        "API overlap",
        "api_similarity",
        "Semantic-static",
        "continuous",
        "low",
        "Overlap of imports, dotted calls, and method-call tokens.",
    ),
    EvidenceSource(
        "API mismatch",
        "api_mismatch_score",
        "Semantic-static",
        "continuous",
        "low",
        "One minus API overlap; a suspicious mismatch heuristic.",
    ),
    EvidenceSource(
        "Identifier overlap",
        "identifier_similarity",
        "Semantic-static",
        "continuous",
        "low",
        "Overlap of non-keyword identifier tokens.",
    ),
    EvidenceSource(
        "Identifier role consistency",
        "identifier_role_similarity",
        "Semantic-static",
        "continuous",
        "medium",
        "Similarity of rough identifier roles such as arguments, assignments, and calls.",
    ),
    EvidenceSource(
        "Data-flow approximation",
        "data_flow_similarity",
        "Semantic-static",
        "continuous",
        "medium",
        "Overlap of approximate assignment/use dependency pairs.",
    ),
    EvidenceSource(
        "Type consistency",
        "type_similarity",
        "Semantic-static",
        "continuous",
        "medium",
        "Overlap of declared type and signature tokens, most informative for Java.",
    ),
    EvidenceSource(
        "Retrieval similarity",
        "retrieval_similarity",
        "Lexical",
        "continuous",
        "low",
        "Text similarity used as a retrieval-style nearest-neighbor proxy.",
    ),
    EvidenceSource(
        "Example execution",
        "execution_passed_example",
        "Dynamic",
        "binary",
        "high",
        "Whether the target program passes the benchmark example tests.",
    ),
    EvidenceSource(
        "Full execution",
        "execution_passed_full",
        "Dynamic",
        "binary",
        "high",
        "Whether the target program passes the full available test suite.",
    ),
]


def taxonomy_rows() -> list[dict[str, str]]:
    """Return all taxonomy rows."""
    return [source.to_dict() for source in EVIDENCE_SOURCES]


def feature_to_category() -> dict[str, str]:
    """Map feature names to evidence categories."""
    return {source.feature: source.category for source in EVIDENCE_SOURCES}


def feature_to_name() -> dict[str, str]:
    """Map feature names to display names."""
    return {source.feature: source.name for source in EVIDENCE_SOURCES}
