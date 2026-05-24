import pytest
from core.action import Action, ActionType, iou


def make_action(bbox, source="dom"):
    return Action(
        action_type=ActionType.CLICK,
        bbox=bbox,
        source=source,
    )


class TestIoU:
    def test_perfect_overlap(self):
        a = make_action((0, 0, 100, 100))
        b = make_action((0, 0, 100, 100))
        assert iou(a, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = make_action((0, 0, 50, 50))
        b = make_action((60, 60, 100, 100))
        assert iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # a and b share a 50x50 region; union is 150x150 - 50x50 = 17500
        a = make_action((0, 0, 100, 100))   # area 10000
        b = make_action((50, 50, 150, 150)) # area 10000, intersection 50x50=2500
        expected = 2500 / (10000 + 10000 - 2500)
        assert iou(a, b) == pytest.approx(expected)

    def test_one_inside_other(self):
        outer = make_action((0, 0, 100, 100))  # area 10000
        inner = make_action((25, 25, 75, 75))  # area 2500, fully inside outer
        expected = 2500 / 10000
        assert iou(outer, inner) == pytest.approx(expected)

    def test_zero_area_action(self):
        a = make_action((50, 50, 50, 50))  # zero area
        b = make_action((0, 0, 100, 100))
        assert iou(a, b) == pytest.approx(0.0)

    def test_symmetry(self):
        a = make_action((0, 0, 80, 60))
        b = make_action((40, 30, 120, 90))
        assert iou(a, b) == pytest.approx(iou(b, a))

    def test_adjacent_no_overlap(self):
        a = make_action((0, 0, 50, 50))
        b = make_action((50, 0, 100, 50))  # touching edge, no area overlap
        assert iou(a, b) == pytest.approx(0.0)
