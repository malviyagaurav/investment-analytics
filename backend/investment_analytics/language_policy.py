from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from .errors import PolicyError


@dataclass(frozen=True)
class LintMatch:
    path: str
    pattern: str
    value: str


DISALLOWED_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bshould\b",
        r"\bmust\b",
        r"\brecommend(?:ed|s|ation|ations)?\b",
        r"\bprefer(?:red|s|ence)?\b",
        r"\bavoid(?:ed|s|ing)?\b",
        r"\bbetter\b",
        r"\bbetter\s+fit\b",
        r"\bsuitable\b",
        r"\bstrong(?:er|est)?\b",
        r"\boutperformance\b",
        r"\boverweight\b",
        r"\bunderweight\b",
        r"\bcheap\b",
        r"\bexpensive\b",
        r"\bswitch(?:ed|es|ing)?\b",
        r"\brebalance(?:d|s|ing)?\b",
        r"\ballocate(?:d|s|ing)?(?:\s+to)?\b",
        r"\ballocation\b",
        r"\bincrease(?:d|s|ing)?\b",
        r"\breduce(?:d|s|ing)?\b",
        r"\bbuy(?:ing|s)?\b",
        r"\bsell(?:ing|s)?\b",
        r"\bbest\b",
        r"\btop\s+(?:fund|etf|stock|bond|asset|pick|candidate|option)s?\b",
        r"\bpick(?:s|ed)?\b",
        r"\btarget[\s_-]*(?:price|allocation|weight)\b",
        r"\bprice[\s_-]*target\b",
        r"\bmodel[\s_-]*portfolio\b",
        r"\boptimized\b",
        r"\bimprovement\b",
    )
)


def _normalized_for_lint(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def _walk_strings(value: Any, path: str = "$") -> Iterable[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield f"{path}.{key}#key", str(key)
            yield from _walk_strings(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{path}[{index}]")


def lint_text_tree(value: Any) -> list[LintMatch]:
    matches: list[LintMatch] = []
    for path, text in _walk_strings(value):
        normalized = _normalized_for_lint(text)
        for pattern in DISALLOWED_PATTERNS:
            if pattern.search(normalized):
                matches.append(LintMatch(path=path, pattern=pattern.pattern, value=text))
    return matches


def assert_language_allowed(value: Any) -> None:
    matches = lint_text_tree(value)
    if matches:
        raise PolicyError(
            "language_policy_violation",
            "Disallowed advisory or preference language detected. Output suppressed.",
            {
                "matches": [
                    {"path": match.path, "pattern": match.pattern, "value": match.value}
                    for match in matches
                ]
            },
        )
