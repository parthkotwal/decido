from core.action import Action
from core.scorer import ScoredAction


def select_best(scored: list[ScoredAction]) -> ScoredAction | None:
    """
    Return the highest-scoring candidate.

    Tie-break (equal score): prefer candidates with higher agreement — an
    action both agents agreed on is safer than one only a single agent proposed
    at the same score.
    """
    if not scored:
        return None
    return max(scored, key=lambda s: (s.score, s.agreement))


def select_top_n(scored: list[ScoredAction], n: int) -> list[ScoredAction]:
    """Return the top-n candidates, already sorted by the scorer."""
    return scored[:n]
