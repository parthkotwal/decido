import pytest
from core.action import Action, ActionType
from core.scorer import score_candidates, ScoredAction


def make_action(bbox, source, confidence=0.8):
    return Action(
        action_type=ActionType.CLICK,
        bbox=bbox,
        source=source,
        confidence=confidence,
    )


class TestScoreCandidates:
    def test_returns_sorted_descending(self):
        candidates = [
            make_action((0, 0, 50, 50), "dom", confidence=0.5),
            make_action((0, 0, 50, 50), "vision", confidence=0.9),
        ]
        scored = score_candidates(candidates)
        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_agreement_boosts_score(self):
        # Two candidates pointing at the exact same region — should get max IoU
        dom = make_action((0, 0, 100, 100), "dom", confidence=0.7)
        vision = make_action((0, 0, 100, 100), "vision", confidence=0.7)
        scored = score_candidates([dom, vision])

        # Both should have agreement = 1.0 (perfect IoU)
        for s in scored:
            assert s.agreement == pytest.approx(1.0)
            # score = 0.5 * 0.7 + 0.5 * 1.0 = 0.85
            assert s.score == pytest.approx(0.85)

    def test_no_agreement_when_single_source(self):
        # Only DOM candidates — no vision to agree with
        candidates = [
            make_action((0, 0, 50, 50), "dom", confidence=0.9),
            make_action((60, 60, 100, 100), "dom", confidence=0.7),
        ]
        scored = score_candidates(candidates)
        for s in scored:
            assert s.agreement == pytest.approx(0.0)

    def test_empty_candidates(self):
        assert score_candidates([]) == []

    def test_score_capped_by_weights(self):
        # Max possible score = 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        dom = make_action((0, 0, 100, 100), "dom", confidence=1.0)
        vision = make_action((0, 0, 100, 100), "vision", confidence=1.0)
        scored = score_candidates([dom, vision])
        for s in scored:
            assert s.score <= 1.0

    def test_non_overlapping_agents_no_agreement(self):
        dom = make_action((0, 0, 50, 50), "dom", confidence=0.8)
        vision = make_action((200, 200, 300, 300), "vision", confidence=0.8)
        scored = score_candidates([dom, vision])
        for s in scored:
            assert s.agreement == pytest.approx(0.0)
