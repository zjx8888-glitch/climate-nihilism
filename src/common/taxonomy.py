"""Official climate opinion taxonomy (from data labels.pdf)."""

from __future__ import annotations

from typing import Dict, List, Optional

TAXONOMY: Dict[str, str] = {
    "Not climate opinion": "Posts that do not relate to climate change.",
    "climate activism": (
        "Urges action on climate change or promotes climate-related projects."
    ),
    "climate anxiety": (
        "Fear or worry about climate change WITHOUT giving up on stopping it. "
        "Distinct from nihilism (hopelessness/futility)."
    ),
    "climate change importance": (
        "States climate change is important; may not call for specific action."
    ),
    "Climate apathy": "Author does not care about climate change.",
    "Climate information": (
        "Informational content (news, disasters, facts) without strong opinion."
    ),
    "Climate nihilism": (
        "Belief that climate change is inevitable/irreversible and action is futile. "
        "Stronger hopelessness than anxiety."
    ),
    "Climate denial": "Claims climate change is not real or not human-caused.",
    "Climate optimism": (
        "Hope that climate is improving or can be improved by people/leaders."
    ),
    "climate policy critique": (
        "Critique of government/political climate policy or politicians."
    ),
    "Climate action critique": (
        "Critique of climate action broadly (corporations, greenwashing, etc.)."
    ),
    "Climate denial critique": "Critique directed at climate deniers.",
    "Climate nihilism critique": (
        "Critique directed at doomism/nihilism — argues against futility."
    ),
    "climate opinion critique": (
        "Critique of other climate opinions not covered above."
    ),
}

CANONICAL_LABELS: List[str] = list(TAXONOMY.keys())

LABEL_ALIASES: Dict[str, str] = {
    "climate information": "Climate information",
    "climate policy critique": "climate policy critique",
    "Climate policy critique": "climate policy critique",
    "climate action critique": "Climate action critique",
    "Climate action critique": "Climate action critique",
    "climate anxiety": "climate anxiety",
    "Climate anxiety": "climate anxiety",
    "climate optimism": "Climate optimism",
    "climate activism critique": "Climate action critique",
    "climate opinion": "climate opinion critique",
    "Climate opinion": "climate opinion critique",
}

PRIORITY_LABELS = frozenset(
    {"Climate nihilism", "climate anxiety", "Climate denial"}
)

RELATED_LABEL_GROUPS: List[List[str]] = [
    ["climate anxiety", "Climate nihilism", "Climate nihilism critique"],
    ["Climate denial", "Climate denial critique"],
    ["Climate nihilism", "Climate nihilism critique", "climate anxiety"],
]

NIHILISM_FOCUS = "Climate nihilism"

_CANONICAL_BY_LOWER: Dict[str, str] = {k.lower(): k for k in CANONICAL_LABELS}
_ALIASES_BY_LOWER: Dict[str, str] = {
    **{k.lower(): v for k, v in LABEL_ALIASES.items()},
    **{v.lower(): v for v in set(LABEL_ALIASES.values())},
}


def _strip_label_text(raw: str) -> str:
    s = str(raw).strip().strip('"').strip()
    if len(s) > 120 or "\n" in s:
        for canonical in CANONICAL_LABELS:
            if canonical in s:
                return canonical
        tail = s.split("\n")[-1].strip().rstrip(".")
        if tail:
            return tail
    return s


def normalize_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = _strip_label_text(raw)
    if not s:
        return None
    if s in TAXONOMY:
        return s
    if s in LABEL_ALIASES:
        return LABEL_ALIASES[s]
    low = s.lower()
    if low in _CANONICAL_BY_LOWER:
        return _CANONICAL_BY_LOWER[low]
    if low in _ALIASES_BY_LOWER:
        return _ALIASES_BY_LOWER[low]
    return None
