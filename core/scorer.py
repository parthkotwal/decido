import re
from dataclasses import dataclass

from core.action import Action, ActionType, iou

_W_AGREEMENT  = 0.60
_W_KEYWORD    = 0.25
_W_COHERENCE  = 0.15

_STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into",
    "fill", "click", "type", "enter", "submit", "find", "page",
    "form", "get", "use", "make", "put", "set", "its", "out", "go",
}

_ROLE_TO_EXPECTED: dict[str, ActionType] = {
    "button":     ActionType.CLICK,
    "link":       ActionType.CLICK,
    "menuitem":   ActionType.CLICK,
    "option":     ActionType.CLICK,
    "checkbox":   ActionType.CLICK,
    "radio":      ActionType.CLICK,
    "tab":        ActionType.CLICK,
    "textbox":    ActionType.TYPE,
    "searchbox":  ActionType.TYPE,
    "combobox":   ActionType.TYPE,
    "spinbutton": ActionType.TYPE,
    "listbox":    ActionType.SELECT,
}


@dataclass
class ScoredAction:
    action: Action
    score: float
    agreement: float
    keyword_match: float
    type_coherence: float


def _best_agreement(action: Action, others: list[Action]) -> float:
    if not others:
        return 0.0
    return max(iou(action, o) for o in others)


def _keyword_match(action: Action, task: str) -> float:
    """Fraction of meaningful task words found in the element name or typed text."""
    task_words = set(re.findall(r'\b[a-z]{3,}\b', task.lower())) - _STOP_WORDS
    if not task_words:
        return 0.5  # no signal — neutral

    haystack = " ".join(filter(None, [
        action.metadata.get("name") or "",
        action.text or "",
    ])).lower()

    if not haystack.strip():
        return 0.5  # no element text to compare (common for vision) — neutral

    matched = sum(1 for w in task_words if w in haystack)
    return matched / len(task_words)


def _type_coherence(action: Action) -> float:
    """1.0 if action type matches the expected type for this element's role, else 0.0.
    Vision candidates have no role metadata — they get a neutral 1.0."""
    role = action.metadata.get("role")
    if not role:
        return 1.0
    expected = _ROLE_TO_EXPECTED.get(role)
    if expected is None:
        return 1.0
    return 1.0 if action.action_type == expected else 0.0


def score_candidates(candidates: list[Action], task: str = "") -> list[ScoredAction]:
    """
    Score a mixed list of DOM and vision candidates.

    Features:
      - agreement (0.60): IoU with the best-matching candidate from the other agent.
            Rewards cases where both agents independently converge on the same region.
      - keyword_match (0.25): fraction of meaningful task words found in the element
            name or text. Grounds the action in what the task is actually asking for.
      - type_coherence (0.15): whether the action type matches the element role
            (e.g. TYPE on a textbox, CLICK on a button). Vision candidates pass
            through at 1.0 since they carry no role metadata.
    """
    dom_candidates    = [a for a in candidates if a.source == "dom"]
    vision_candidates = [a for a in candidates if a.source == "vision"]

    scored: list[ScoredAction] = []
    for action in candidates:
        others         = vision_candidates if action.source == "dom" else dom_candidates
        agreement      = _best_agreement(action, others)
        keyword        = _keyword_match(action, task)
        coherence      = _type_coherence(action)
        score          = _W_AGREEMENT * agreement + _W_KEYWORD * keyword + _W_COHERENCE * coherence

        scored.append(ScoredAction(
            action=action,
            score=score,
            agreement=agreement,
            keyword_match=keyword,
            type_coherence=coherence,
        ))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored
