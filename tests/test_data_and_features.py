"""Tests for dataset construction and static features."""

from evicode.data import ProgramRecord, build_pairs, parse_problem_id
from evicode.execution.runner import select_test
from evicode.features import static_features
from evicode.taxonomy import feature_to_category, taxonomy_rows


def make_record(problem_id: int, language: str, body: str) -> ProgramRecord:
    """Create a small synthetic program record."""
    return ProgramRecord(
        problem_id=problem_id,
        language=language,
        task_id=f"HumanEval/{problem_id}",
        prompt="",
        declaration="",
        canonical_solution=body,
        test=f"assert candidate_{problem_id}() == {problem_id}",
        example_test=f"assert candidate_{problem_id}() == {problem_id}",
    )


def test_parse_problem_id() -> None:
    """HumanEval task ids should map to integer problem ids."""
    assert parse_problem_id("HumanEval/42") == 42


def test_build_pairs_creates_positive_mismatch_and_mutation() -> None:
    """Directed pair construction should create one positive and two negatives per direction."""
    records = {
        "python": [
            make_record(0, "python", "def candidate_0():\n    return 0\n"),
            make_record(1, "python", "def candidate_1():\n    return 1\n"),
        ],
        "js": [
            make_record(0, "js", "function candidate_0() { return 0; }\n"),
            make_record(1, "js", "function candidate_1() { return 1; }\n"),
        ],
    }
    examples = build_pairs(records, ["python", "js"], seed=7)
    assert len(examples) == 12
    assert sum(example.label for example in examples) == 4
    assert {example.negative_type for example in examples} == {"none", "mismatch", "mutation"}


def test_static_features_detect_invalid_python_target() -> None:
    """Parser-backed syntax evidence should reject invalid target code."""
    features = static_features(
        source_code="def f():\n    return 1\n",
        target_code="def f(:\n    return 1\n",
        source_language="python",
        target_language="python",
    )
    assert features["source_syntax_valid"] == 1.0
    assert features["target_syntax_valid"] == 0.0
    assert features["syntax_proxy"] == 0.0
    assert "token_jaccard" in features
    assert "data_flow_similarity" in features
    assert "type_similarity" in features


def test_select_test_budget() -> None:
    """Execution budgets should choose the expected test body."""
    assert select_test("full", "example", "none") == ""
    assert select_test("full", "example", "example") == "example"
    assert select_test("full", "example", "full") == "full"


def test_evidence_taxonomy_has_expected_groups() -> None:
    """The explicit taxonomy should expose the main evidence groups."""
    categories = {row["category"] for row in taxonomy_rows()}
    assert {
        "Weak-proxy",
        "Normalized-program",
        "Normalized-control",
        "Normalized-structure",
        "Normalized-operator",
        "Normalized-identifier",
        "Normalized-dataflow",
        "Dynamic",
    } <= categories
    mapping = feature_to_category()
    assert mapping["token_jaccard"] == "Weak-proxy"
    assert mapping["ln_operator_family_similarity"] == "Normalized-operator"
    assert mapping["execution_passed_full"] == "Dynamic"
