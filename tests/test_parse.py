"""
Tests for proposal parsing in both agents.
These are pure functions — no Playwright, no API calls needed.
"""

import pytest
from agents.vision_agent import _parse_proposal
from agents.dom_agent import _parse_response


class TestVisionParseProposal:
    def test_valid_click(self):
        action = _parse_proposal({
            "action": "click",
            "bbox": [10, 20, 100, 80],
        })
        assert action is not None
        assert action.action_type.value == "click"
        assert action.bbox == (10.0, 20.0, 100.0, 80.0)
        assert action.text is None

    def test_valid_type_with_text(self):
        action = _parse_proposal({
            "action": "type",
            "bbox": [0, 0, 200, 40],
            "text": "hello@example.com",
        })
        assert action is not None
        assert action.text == "hello@example.com"

    def test_type_without_text_rejected(self):
        assert _parse_proposal({
            "action": "type",
            "bbox": [0, 0, 200, 40],
        }) is None

    def test_type_with_empty_text_rejected(self):
        assert _parse_proposal({
            "action": "type",
            "bbox": [0, 0, 200, 40],
            "text": "",
        }) is None

    def test_select_without_text_rejected(self):
        assert _parse_proposal({
            "action": "select",
            "bbox": [0, 0, 200, 40],
        }) is None

    def test_invalid_action_type_rejected(self):
        assert _parse_proposal({
            "action": "explode",
            "bbox": [0, 0, 100, 100],
        }) is None

    def test_zero_area_bbox_rejected(self):
        assert _parse_proposal({
            "action": "click",
            "bbox": [50, 50, 50, 50],
        }) is None

    def test_inverted_bbox_rejected(self):
        assert _parse_proposal({
            "action": "click",
            "bbox": [100, 100, 10, 10],
        }) is None

    def test_missing_bbox_rejected(self):
        assert _parse_proposal({"action": "click"}) is None

    def test_click_with_empty_text_coerced_to_none(self):
        action = _parse_proposal({
            "action": "click",
            "bbox": [0, 0, 100, 100],
            "text": "",
        })
        assert action is not None
        assert action.text is None


class TestDomParseResponse:
    def test_valid_json_array(self):
        raw = '[{"node_id": 1, "action": "click"}]'
        result = _parse_response(raw)
        assert len(result) == 1
        assert result[0]["node_id"] == 1

    def test_markdown_fenced_json(self):
        raw = '```json\n[{"node_id": 2, "action": "click"}]\n```'
        result = _parse_response(raw)
        assert len(result) == 1
        assert result[0]["node_id"] == 2

    def test_invalid_json_returns_empty(self):
        assert _parse_response("not json at all") == []

    def test_non_list_json_returns_empty(self):
        assert _parse_response('{"node_id": 1}') == []

    def test_empty_array(self):
        assert _parse_response("[]") == []

    def test_multiple_proposals(self):
        raw = '[{"node_id": 1, "action": "click"}, {"node_id": 2, "action": "type", "text": "hi"}]'
        result = _parse_response(raw)
        assert len(result) == 2
