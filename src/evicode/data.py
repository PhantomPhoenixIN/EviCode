"""Dataset records and HumanEval-X pair construction."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ProgramRecord:
    """One HumanEval-X program record."""

    problem_id: int
    language: str
    task_id: str
    prompt: str
    declaration: str
    canonical_solution: str
    test: str
    example_test: str

    @property
    def full_code(self) -> str:
        """Return executable candidate code without tests."""
        return f"{self.declaration}{self.canonical_solution}"


@dataclass(frozen=True)
class VerificationExample:
    """One source-target verification example."""

    example_id: str
    problem_id: int
    source_language: str
    target_language: str
    source_code: str
    target_code: str
    target_test: str
    target_example_test: str
    label: int
    negative_type: str

    def to_json(self) -> dict[str, Any]:
        """Serialize example to JSON-compatible data."""
        return asdict(self)


def parse_problem_id(task_id: str) -> int:
    """Parse integer HumanEval-X problem id from a task id."""
    return int(task_id.split("/")[-1])


def mutate_code(code: str, language: str, rng: random.Random) -> str:
    """Create a controlled likely-incorrect mutation while preserving syntax when possible."""
    replacements = [
        (r"\breturn true\b", "return false"),
        (r"\breturn false\b", "return true"),
        (r"\breturn True\b", "return False"),
        (r"\breturn False\b", "return True"),
        (r"===", "!=="),
        (r"==", "!="),
        (r"!=", "=="),
        (r"<=", "<"),
        (r">=", ">"),
        (r"<", "<="),
        (r">", ">="),
        (r"\+", "-"),
    ]
    shuffled = replacements[:]
    rng.shuffle(shuffled)
    for pattern, replacement in shuffled:
        mutated, count = re.subn(pattern, replacement, code, count=1)
        if count:
            return mutated
    if language == "python":
        return code + "\n# mutation-noop\n"
    if language == "java":
        return code.replace("return ", "return ", 1)
    return code + "\n// mutation-noop\n"


def build_pairs(records: dict[str, list[ProgramRecord]], languages: list[str], seed: int) -> list[VerificationExample]:
    """Build positive and controlled negative directed translation verification pairs."""
    rng = random.Random(seed)
    examples: list[VerificationExample] = []
    by_lang_problem = {
        lang: {record.problem_id: record for record in records[lang]} for lang in languages
    }
    problem_ids = sorted(set.intersection(*(set(by_lang_problem[lang]) for lang in languages)))
    for problem_id in problem_ids:
        for source_language in languages:
            for target_language in languages:
                if source_language == target_language:
                    continue
                source = by_lang_problem[source_language][problem_id]
                target = by_lang_problem[target_language][problem_id]
                base = f"{problem_id}_{source_language}_to_{target_language}"
                examples.append(
                    VerificationExample(
                        example_id=f"{base}_positive",
                        problem_id=problem_id,
                        source_language=source_language,
                        target_language=target_language,
                        source_code=source.full_code,
                        target_code=target.full_code,
                        target_test=target.test,
                        target_example_test=target.example_test,
                        label=1,
                        negative_type="none",
                    )
                )
                mismatch_id = problem_ids[(problem_ids.index(problem_id) + 1) % len(problem_ids)]
                mismatch = by_lang_problem[target_language][mismatch_id]
                examples.append(
                    VerificationExample(
                        example_id=f"{base}_mismatch",
                        problem_id=problem_id,
                        source_language=source_language,
                        target_language=target_language,
                        source_code=source.full_code,
                        target_code=mismatch.full_code,
                        target_test=target.test,
                        target_example_test=target.example_test,
                        label=0,
                        negative_type="mismatch",
                    )
                )
                examples.append(
                    VerificationExample(
                        example_id=f"{base}_mutation",
                        problem_id=problem_id,
                        source_language=source_language,
                        target_language=target_language,
                        source_code=source.full_code,
                        target_code=mutate_code(target.full_code, target_language, rng),
                        target_test=target.test,
                        target_example_test=target.example_test,
                        label=0,
                        negative_type="mutation",
                    )
                )
    return examples
