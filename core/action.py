from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    SELECT = "select"
    HOVER = "hover"


# (x1, y1, x2, y2) in page coordinates (pixels)
BBox = tuple[float, float, float, float]


@dataclass
class Action:
    action_type: ActionType
    bbox: BBox
    source: str                     # "dom" or "vision"
    confidence: float               # agent's self-reported confidence [0, 1]
    text: Optional[str] = None      # payload for "type" and "select"
    element_ref: Optional[str] = None  # axtree node ID, DOM agent only
    scroll_direction: Optional[str] = None  # "up" / "down", scroll only
    scroll_amount: Optional[int] = None     # pixels, scroll only
    metadata: dict = field(default_factory=dict)  # agent-specific extras

    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Action, b: Action) -> float:
    """Intersection-over-union of two actions' bounding boxes."""
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area() + b.area() - intersection

    return intersection / union if union > 0 else 0.0
