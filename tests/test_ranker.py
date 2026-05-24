import pytest
from core.action import Action, ActionType
from core.scorer import ScoredAction
from core.ranker import select_best, select_top_n


def make_scored(score, agreement=0.0, source="dom"):
    action = Action(
        action_type=ActionType.CLICK,
        bbox=(0, 0, 50, 50),
        source=source,
    )
    return ScoredAction(
        action=action,
        score=score,
        agreement=agreement,
        keyword_match=0.5,
        type_coherence=1.0,
    )


class TestSelectBest:
    def test_returns_highest_score(self):
        candidates = [make_scored(0.3), make_scored(0.8), make_scored(0.5)]
        best = select_best(candidates)
        assert best.score == pytest.approx(0.8)

    def test_empty_returns_none(self):
        assert select_best([]) is None

    def test_tie_broken_by_agreement(self):
        low_agree = make_scored(0.7, agreement=0.2)
        high_agree = make_scored(0.7, agreement=0.9)
        best = select_best([low_agree, high_agree])
        assert best.agreement == pytest.approx(0.9)

    def test_single_candidate(self):
        only = make_scored(0.6)
        assert select_best([only]) is only


class TestSelectTopN:
    def test_returns_top_n(self):
        candidates = [make_scored(s) for s in [0.9, 0.7, 0.5, 0.3]]
        top2 = select_top_n(candidates, 2)
        assert len(top2) == 2
        assert top2[0].score == pytest.approx(0.9)
        assert top2[1].score == pytest.approx(0.7)

    def test_n_larger_than_list(self):
        candidates = [make_scored(0.5), make_scored(0.3)]
        assert select_top_n(candidates, 10) == candidates

    def test_empty(self):
        assert select_top_n([], 3) == []
