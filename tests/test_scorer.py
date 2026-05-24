import pytest
from core.action import Action, ActionType
from core.scorer import score_candidates, ScoredAction, _W_AGREEMENT, _W_KEYWORD, _W_COHERENCE


def make_action(bbox, source, metadata=None, text=None):
    return Action(
        action_type=ActionType.CLICK,
        bbox=bbox,
        source=source,
        metadata=metadata or {},
        text=text,
    )


TASK = "fill in the email field"


class TestScoreCandidates:
    def test_returns_sorted_descending(self):
        candidates = [
            make_action((0, 0, 50, 50), "dom"),
            make_action((0, 0, 50, 50), "vision"),
        ]
        scored = score_candidates(candidates, TASK)
        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_agreement_boosts_score(self):
        # Perfect overlap — both agents agree on the exact same region
        dom    = make_action((0, 0, 100, 100), "dom")
        vision = make_action((0, 0, 100, 100), "vision")
        scored = score_candidates([dom, vision], TASK)

        for s in scored:
            assert s.agreement == pytest.approx(1.0)

        dom_scored    = next(s for s in scored if s.action.source == "dom")
        vision_scored = next(s for s in scored if s.action.source == "vision")

        # DOM: no element name → keyword=None → redistributed weights (agreement+keyword → agreement)
        dom_expected = (_W_AGREEMENT + _W_KEYWORD) * 1.0 + _W_COHERENCE * 1.0
        assert dom_scored.score == pytest.approx(dom_expected)

        # Vision: no name/text → keyword=None → redistributed weights
        vision_expected = (_W_AGREEMENT + _W_KEYWORD) * 1.0 + _W_COHERENCE * 1.0
        assert vision_scored.score == pytest.approx(vision_expected)

    def test_no_agreement_when_single_source(self):
        candidates = [
            make_action((0, 0, 50, 50), "dom"),
            make_action((60, 60, 100, 100), "dom"),
        ]
        scored = score_candidates(candidates, TASK)
        for s in scored:
            assert s.agreement == pytest.approx(0.0)

    def test_empty_candidates(self):
        assert score_candidates([], TASK) == []

    def test_score_capped_at_one(self):
        dom    = make_action((0, 0, 100, 100), "dom")
        vision = make_action((0, 0, 100, 100), "vision")
        scored = score_candidates([dom, vision], TASK)
        for s in scored:
            assert s.score <= 1.0

    def test_non_overlapping_agents_no_agreement(self):
        dom    = make_action((0, 0, 50, 50), "dom")
        vision = make_action((200, 200, 300, 300), "vision")
        scored = score_candidates([dom, vision], TASK)
        for s in scored:
            assert s.agreement == pytest.approx(0.0)

    def test_keyword_match_boosts_dom_candidate(self):
        matching = Action(
            action_type=ActionType.TYPE,
            bbox=(0, 0, 100, 30),
            source="dom",
            metadata={"role": "textbox", "name": "email"},
        )
        non_matching = Action(
            action_type=ActionType.TYPE,
            bbox=(200, 200, 300, 230),
            source="dom",
            metadata={"role": "textbox", "name": "phone"},
        )
        scored = score_candidates([matching, non_matching], "fill in the email field")
        assert scored[0].action.metadata["name"] == "email"
        assert scored[0].keyword_match > scored[1].keyword_match

    def test_type_coherence_penalises_wrong_action_type(self):
        coherent = Action(
            action_type=ActionType.TYPE,
            bbox=(0, 0, 100, 30),
            source="dom",
            metadata={"role": "textbox", "name": "email"},
        )
        incoherent = Action(
            action_type=ActionType.TYPE,
            bbox=(0, 50, 100, 80),
            source="dom",
            metadata={"role": "button", "name": "submit"},
        )
        scored = score_candidates([coherent, incoherent], "type email")
        coherent_scored   = next(s for s in scored if s.action.metadata["name"] == "email")
        incoherent_scored = next(s for s in scored if s.action.metadata["name"] == "submit")
        assert coherent_scored.type_coherence   == pytest.approx(1.0)
        assert incoherent_scored.type_coherence == pytest.approx(0.0)

    def test_vision_candidate_gets_neutral_coherence(self):
        vision = make_action((0, 0, 100, 30), "vision")
        scored = score_candidates([vision], TASK)
        assert scored[0].type_coherence == pytest.approx(1.0)
