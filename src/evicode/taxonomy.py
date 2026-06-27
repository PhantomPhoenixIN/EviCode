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
        "Weak-proxy",
        "continuous",
        "low",
        "Jaccard overlap over raw lexical tokens; retained as a weak cross-language proxy.",
    ),
    EvidenceSource(
        "Edit similarity",
        "edit_similarity",
        "Weak-proxy",
        "continuous",
        "low",
        "Normalized raw text similarity; retained as a weak cross-language proxy.",
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
        "Weak-proxy",
        "continuous",
        "medium",
        "Cosine similarity over language-specific parser node types; retained only as a weak proxy.",
    ),
    EvidenceSource(
        "AST depth similarity",
        "ast_depth_similarity",
        "Normalized-structure",
        "continuous",
        "medium",
        "Similarity of maximum parse-tree depth; an approximation, not tree edit distance.",
    ),
    EvidenceSource(
        "AST shape similarity",
        "ast_shape_similarity",
        "Normalized-structure",
        "continuous",
        "medium",
        "Similarity of depth histograms from parse trees.",
    ),
    EvidenceSource(
        "Control flow",
        "control_flow_similarity",
        "Normalized-control",
        "continuous",
        "low",
        "Similarity over branch, loop, return, and exception keyword counts.",
    ),
    EvidenceSource(
        "Nesting depth",
        "nesting_depth_similarity",
        "Normalized-control",
        "continuous",
        "low",
        "Similarity of approximate brace/indentation nesting depth.",
    ),
    EvidenceSource(
        "Condition/operator patterns",
        "operator_pattern_similarity",
        "Normalized-operator",
        "continuous",
        "low",
        "Similarity of conditional and boolean/comparison operator patterns.",
    ),
    EvidenceSource(
        "API overlap",
        "api_similarity",
        "Weak-proxy",
        "continuous",
        "low",
        "Overlap of language-specific imports, dotted calls, and method-call tokens; weak across languages.",
    ),
    EvidenceSource(
        "API mismatch",
        "api_mismatch_score",
        "Weak-proxy",
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
        "Weak-proxy",
        "continuous",
        "low",
        "Text similarity used as a retrieval-style nearest-neighbor proxy.",
    ),
    EvidenceSource("LN syntax validity", "ln_syntax_both_valid", "Normalized-program", "binary", "low", "Whether both programs parse in their own languages."),
    EvidenceSource("LN CFG node count", "ln_cfg_nodes_similarity", "Normalized-control", "continuous", "low", "Similarity of normalized CFG node-count proxy."),
    EvidenceSource("LN CFG edge count", "ln_cfg_edges_similarity", "Normalized-control", "continuous", "low", "Similarity of normalized CFG edge-count proxy."),
    EvidenceSource("LN cyclomatic complexity", "ln_cyclomatic_complexity_similarity", "Normalized-control", "continuous", "low", "Similarity of decision-count based cyclomatic complexity."),
    EvidenceSource("LN branch count", "ln_branch_count_similarity", "Normalized-control", "continuous", "low", "Similarity of branch counts extracted independently from both languages."),
    EvidenceSource("LN loop count", "ln_loop_count_similarity", "Normalized-control", "continuous", "low", "Similarity of loop counts extracted independently from both languages."),
    EvidenceSource("LN return count", "ln_return_count_similarity", "Normalized-control", "continuous", "low", "Similarity of return statement counts."),
    EvidenceSource("LN exception count", "ln_exception_count_similarity", "Normalized-control", "continuous", "low", "Similarity of exception-handling counts."),
    EvidenceSource("LN function count", "ln_function_count_similarity", "Normalized-structure", "continuous", "low", "Similarity of function or method counts."),
    EvidenceSource("LN class count", "ln_class_count_similarity", "Normalized-structure", "continuous", "low", "Similarity of class/interface counts."),
    EvidenceSource("LN max AST depth", "ln_max_ast_depth_similarity", "Normalized-structure", "continuous", "medium", "Similarity of maximum parser-tree depth without comparing node labels."),
    EvidenceSource("LN average tree depth", "ln_avg_tree_depth_similarity", "Normalized-structure", "continuous", "medium", "Similarity of average parser-tree depth."),
    EvidenceSource("LN branching factor", "ln_branching_factor_similarity", "Normalized-structure", "continuous", "medium", "Similarity of parser-tree branching factor proxy."),
    EvidenceSource("LN nesting depth", "ln_nesting_depth_similarity", "Normalized-control", "continuous", "low", "Similarity of brace/indentation nesting depth."),
    EvidenceSource("LN call count", "ln_call_count_similarity", "Normalized-call", "continuous", "low", "Similarity of function and method call counts."),
    EvidenceSource("LN identifier count", "ln_identifier_count_similarity", "Normalized-identifier", "continuous", "low", "Similarity of non-keyword identifier counts."),
    EvidenceSource("LN identifier entropy", "ln_identifier_entropy_similarity", "Normalized-identifier", "continuous", "low", "Similarity of normalized identifier entropy."),
    EvidenceSource("LN identifier role count", "ln_identifier_role_count_similarity", "Normalized-identifier", "continuous", "medium", "Similarity of extracted identifier-role counts."),
    EvidenceSource("LN def-use count", "ln_def_use_count_similarity", "Normalized-dataflow", "continuous", "medium", "Similarity of approximate definition-use pair counts."),
    EvidenceSource("LN read-write ratio", "ln_read_write_ratio_similarity", "Normalized-dataflow", "continuous", "medium", "Similarity of approximate read/write ratios."),
    EvidenceSource("LN assignment density", "ln_assignment_density_similarity", "Normalized-dataflow", "continuous", "low", "Similarity of assignments per non-empty line."),
    EvidenceSource("LN statement density", "ln_statement_density_similarity", "Normalized-structure", "continuous", "low", "Similarity of normalized statement density."),
    EvidenceSource("LN expression density", "ln_expression_density_similarity", "Normalized-structure", "continuous", "low", "Similarity of normalized expression density."),
    EvidenceSource("LN operator families", "ln_operator_family_similarity", "Normalized-operator", "continuous", "low", "Cosine similarity of arithmetic, comparison, logical, and assignment operator families."),
    EvidenceSource("LN statement distribution", "ln_statement_distribution_similarity", "Normalized-structure", "continuous", "low", "Cosine similarity of language-normalized statement distributions."),
    EvidenceSource("LN expression distribution", "ln_expression_distribution_similarity", "Normalized-structure", "continuous", "low", "Cosine similarity of language-normalized expression distributions."),
    EvidenceSource("LN identifier-role distribution", "ln_identifier_role_distribution_similarity", "Normalized-identifier", "continuous", "medium", "Cosine similarity of declaration, assignment, and call role distributions."),
    EvidenceSource("LN control profile", "ln_control_profile_similarity", "Normalized-control", "continuous", "low", "Cosine similarity over branch, loop, return, and exception counts."),
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


def evidence_family(feature: str, category: str) -> str:
    """Return the higher-level evidence family for a feature."""
    if category == "Dynamic" or "syntax" in feature or "valid" in feature:
        return "Reliability"
    if "complexity" in feature or "density" in feature or "depth" in feature or "branching_factor" in feature:
        return "Complexity"
    if (
        "operator" in feature
        or "return" in feature
        or "exception" in feature
        or "call" in feature
        or "identifier" in feature
        or "data_flow" in feature
        or "def_use" in feature
        or "read_write" in feature
        or "api" in feature
        or "type" in feature
    ):
        return "Behavioral"
    return "Structural"


def feature_to_family() -> dict[str, str]:
    """Map feature names to high-level evidence families."""
    return {source.feature: evidence_family(source.feature, source.category) for source in EVIDENCE_SOURCES}
