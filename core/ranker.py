from core.action import Action, ActionType
from core.scorer import ScoredAction

_ACTION_PRIORITY = {
    ActionType.TYPE: 3,
    ActionType.SELECT: 3,
    ActionType.CLICK: 2,
    ActionType.SCROLL: 1,
    ActionType.HOVER: 0,
}


def select_best(scored: list[ScoredAction]) -> ScoredAction | None:
    """
    Return the highest-scoring candidate.

    Tie-break chain (when scores are equal):
      1. Higher agreement — an action both agents agreed on is safer
      2. Action type priority — type/select over click (typing already
         focuses the element, so a bare click on an input is redundant)
    """
    if not scored:
        return None
    return max(scored, key=lambda s: (
        s.score,
        s.agreement,
        _ACTION_PRIORITY.get(s.action.action_type, 0),
    ))


def select_top_n(scored: list[ScoredAction], n: int) -> list[ScoredAction]:
    """Return the top-n candidates, already sorted by the scorer."""
    return scored[:n]
