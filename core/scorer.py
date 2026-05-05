from dataclasses import dataclass

from core.action import Action, iou

# v1 weights — tunable, must sum to 1.0
_W_CONFIDENCE = 0.5
_W_AGREEMENT = 0.5


@dataclass
class ScoredAction:
    action: Action
    score: float
    confidence: float   # agent's self-reported value, passed through for logging
    agreement: float    # IoU with best-matching candidate from the other source


def _best_agreement(action: Action, others: list[Action]) -> float:
    """Max IoU between action and any candidate from the opposite source."""
    if not others:
        return 0.0
    return max(iou(action, o) for o in others)


def score_candidates(candidates: list[Action]) -> list[ScoredAction]:
    """
    Score a mixed list of DOM and vision candidates.

    Each candidate is scored on two features:
      - confidence:  the agent's self-reported confidence [0, 1]
      - agreement:   max IoU with any candidate from the opposite agent [0, 1]

    Agreement rewards candidates that both agents independently pointed at the
    same region — the core signal that makes the dual-agent setup worthwhile.

    Returns ScoredActions sorted by score descending.
    """
    dom_candidates = [a for a in candidates if a.source == "dom"]
    vision_candidates = [a for a in candidates if a.source == "vision"]

    scored: list[ScoredAction] = []
    for action in candidates:
        others = vision_candidates if action.source == "dom" else dom_candidates
        agreement = _best_agreement(action, others)
        score = _W_CONFIDENCE * action.confidence + _W_AGREEMENT * agreement
        scored.append(
            ScoredAction(
                action=action,
                score=score,
                confidence=action.confidence,
                agreement=agreement,
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored
