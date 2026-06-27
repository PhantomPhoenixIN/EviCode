"""Static evidence feature extraction."""

from __future__ import annotations

import ast
import keyword
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable

from tree_sitter import Language, Parser
import tree_sitter_java
import tree_sitter_javascript
import tree_sitter_python


CONTROL_TERMS = {
    "if": ["if", "else", "elif", "switch", "case"],
    "loop": ["for", "while", "do"],
    "return": ["return"],
    "exception": ["try", "catch", "except", "throw", "throws"],
}

TREE_SITTER_LANGUAGES = {
    "python": Language(tree_sitter_python.language()),
    "java": Language(tree_sitter_java.language()),
    "js": Language(tree_sitter_javascript.language()),
    "javascript": Language(tree_sitter_javascript.language()),
}


def identifiers(code: str) -> set[str]:
    """Extract rough identifier set."""
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", code))
    stop = set(keyword.kwlist) | {
        "const",
        "let",
        "var",
        "function",
        "class",
        "public",
        "static",
        "void",
        "new",
        "return",
        "true",
        "false",
        "True",
        "False",
        "import",
        "from",
        "java",
        "util",
    }
    return {word.lower() for word in words if word not in stop and len(word) > 1}


def api_tokens(code: str) -> set[str]:
    """Extract rough API/library tokens."""
    dotted = re.findall(r"\b([A-Z][A-Za-z0-9_]*|[a-z][A-Za-z0-9_]*)\s*\.", code)
    imports = re.findall(r"\b(?:import|from|require)\s+([A-Za-z0-9_.*]+)", code)
    methods = re.findall(r"\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
    return {token.lower() for token in dotted + imports + methods}


def counter_for_terms(code: str, terms: Iterable[str]) -> int:
    """Count term occurrences as keywords."""
    return sum(len(re.findall(rf"\b{re.escape(term)}\b", code)) for term in terms)


def control_vector(code: str) -> dict[str, int]:
    """Extract a control-flow proxy vector."""
    return {name: counter_for_terms(code, terms) for name, terms in CONTROL_TERMS.items()}


def python_ast_counts(code: str) -> Counter[str]:
    """Extract Python AST node counts, or an empty counter if parsing fails."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()
    return Counter(type(node).__name__ for node in ast.walk(tree))


def tree_sitter_counts(code: str, language: str) -> tuple[bool, Counter[str], int, Counter[str]]:
    """Parse code and return validity, node counts, max depth, and depth histogram."""
    ts_language = TREE_SITTER_LANGUAGES.get(language.lower())
    if ts_language is None:
        return False, Counter(), 0, Counter()
    parser = Parser(ts_language)
    tree = parser.parse(code.encode("utf-8", errors="ignore"))
    counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    max_depth = 0
    stack = [(tree.root_node, 0)]
    while stack:
        node, depth = stack.pop()
        counts[node.type] += 1
        depth_counts[str(depth)] += 1
        max_depth = max(max_depth, depth)
        stack.extend((child, depth + 1) for child in node.children)
    return not tree.root_node.has_error, counts, max_depth, depth_counts


def lexical_counts(code: str) -> Counter[str]:
    """Extract a language-agnostic lexical structure counter."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|==|!=|<=|>=|[{}()[\\];,+\\-*/%<>]", code)
    return Counter(token.lower() for token in tokens)


def token_set(code: str) -> set[str]:
    """Extract language-agnostic lexical token set."""
    return set(lexical_counts(code))


def cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    """Cosine similarity between sparse counters."""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a[key] * b[key] for key in keys)
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def scalar_similarity(a: int | float, b: int | float) -> float:
    """Return a bounded similarity for two non-negative scalar values."""
    return min(float(a), float(b)) / max(float(a), float(b), 1.0)


def nesting_depth(code: str, language: str) -> int:
    """Approximate control nesting depth from braces or Python indentation."""
    if language.lower() == "python":
        indents = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped:
                indents.append((len(line) - len(line.lstrip(" "))) // 4)
        return max(indents, default=0)
    depth = 0
    max_depth = 0
    for char in code:
        if char == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == "}":
            depth = max(depth - 1, 0)
    return max_depth


def operator_counter(code: str) -> Counter[str]:
    """Count conditional, boolean, and comparison operators."""
    patterns = {
        "eq": r"==|===",
        "neq": r"!=|!==",
        "le": r"<=",
        "ge": r">=",
        "lt": r"(?<!<)<(?![=<])",
        "gt": r"(?<!>)>(?![=>])",
        "and": r"\band\b|&&",
        "or": r"\bor\b|\|\|",
        "not": r"\bnot\b|!",
    }
    return Counter({name: len(re.findall(pattern, code)) for name, pattern in patterns.items()})


def call_tokens(code: str) -> set[str]:
    """Extract rough function and method call tokens."""
    return {token.lower() for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)}


def identifier_roles(code: str) -> set[str]:
    """Extract approximate identifier roles."""
    roles = set()
    for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=", code):
        roles.add(f"assign:{token.lower()}")
    for token in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", code):
        roles.add(f"call:{token.lower()}")
    for token in re.findall(r"\b(?:def|function|public|private|protected)\s+[A-Za-z0-9_<>,\\[\\]]*\s*([A-Za-z_][A-Za-z0-9_]*)", code):
        roles.add(f"decl:{token.lower()}")
    return roles


def data_flow_pairs(code: str) -> set[str]:
    """Extract approximate assignment/use dependency pairs."""
    pairs = set()
    for lhs, rhs in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^;\n]+)", code):
        for used in identifiers(rhs):
            pairs.add(f"{lhs.lower()}->{used}")
    return pairs


def type_tokens(code: str) -> set[str]:
    """Extract approximate declared type and signature tokens."""
    java_types = r"(?:int|long|double|float|boolean|Boolean|String|char|void|List|ArrayList|Map|Set|HashMap|HashSet|Integer|Double)"
    tokens = set(re.findall(rf"\b{java_types}\b", code))
    py_types = re.findall(r":\s*([A-Za-z_][A-Za-z0-9_\\[\\]]*)", code) + re.findall(
        r"->\s*([A-Za-z_][A-Za-z0-9_\\[\\]]*)", code
    )
    return {token.lower() for token in tokens | set(py_types)}


def static_features(source_code: str, target_code: str, source_language: str, target_language: str) -> dict[str, float]:
    """Extract static evidence features for one code pair."""
    source_ids = identifiers(source_code)
    target_ids = identifiers(target_code)
    source_api = api_tokens(source_code)
    target_api = api_tokens(target_code)
    source_control = Counter(control_vector(source_code))
    target_control = Counter(control_vector(target_code))
    source_valid, source_ast, source_depth, source_depth_counts = tree_sitter_counts(source_code, source_language)
    target_valid, target_ast, target_depth, target_depth_counts = tree_sitter_counts(target_code, target_language)
    ast_sim = cosine_counter(source_ast, target_ast)
    if ast_sim == 0.0 and source_language == "python" and target_language == "python":
        ast_sim = cosine_counter(python_ast_counts(source_code), python_ast_counts(target_code))
    if ast_sim == 0.0:
        ast_sim = cosine_counter(lexical_counts(source_code), lexical_counts(target_code))
    return {
        "syntax_proxy": float(source_valid and target_valid),
        "source_syntax_valid": float(source_valid),
        "target_syntax_valid": float(target_valid),
        "token_jaccard": jaccard(token_set(source_code), token_set(target_code)),
        "edit_similarity": SequenceMatcher(None, source_code, target_code).ratio(),
        "ast_similarity": ast_sim,
        "ast_depth_similarity": scalar_similarity(source_depth, target_depth),
        "ast_shape_similarity": cosine_counter(source_depth_counts, target_depth_counts),
        "control_flow_similarity": cosine_counter(source_control, target_control),
        "branch_count_similarity": scalar_similarity(source_control["if"], target_control["if"]),
        "loop_count_similarity": scalar_similarity(source_control["loop"], target_control["loop"]),
        "return_count_similarity": scalar_similarity(source_control["return"], target_control["return"]),
        "nesting_depth_similarity": scalar_similarity(
            nesting_depth(source_code, source_language), nesting_depth(target_code, target_language)
        ),
        "operator_pattern_similarity": cosine_counter(operator_counter(source_code), operator_counter(target_code)),
        "api_similarity": jaccard(source_api, target_api),
        "call_similarity": jaccard(call_tokens(source_code), call_tokens(target_code)),
        "api_mismatch_score": 1.0 - jaccard(source_api | call_tokens(source_code), target_api | call_tokens(target_code)),
        "identifier_similarity": jaccard(source_ids, target_ids),
        "identifier_role_similarity": jaccard(identifier_roles(source_code), identifier_roles(target_code)),
        "data_flow_similarity": jaccard(data_flow_pairs(source_code), data_flow_pairs(target_code)),
        "type_similarity": jaccard(type_tokens(source_code), type_tokens(target_code)),
        "retrieval_similarity": SequenceMatcher(None, source_code, target_code).ratio(),
        "source_length": float(len(source_code)),
        "target_length": float(len(target_code)),
        "length_ratio": min(len(source_code), len(target_code)) / max(len(source_code), len(target_code), 1),
    }
