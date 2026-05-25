import logging
import modal
from playwright.async_api import Page

from core.action import Action, ActionType, BBox

logger = logging.getLogger(__name__)

_VisionModel = modal.Cls.from_name("decido-vision", "VisionModel")

_ACTION_TYPES = {a.value for a in ActionType}


async def propose_actions(
    page: Page,
    task: str,
    max_candidates: int = 3,
    memory_context: str | None = None,
) -> list[Action]:
    """
    Take a screenshot of the current page, send it to Qwen2.5-VL on Modal,
    and return up to max_candidates Action proposals.

    Returns an empty list (rather than raising) on any failure so the pipeline
    can continue with DOM-only candidates.
    """
    try:
        screenshot_bytes: bytes = await page.screenshot(type="png", full_page=False)
    except Exception:
        logger.exception("vision agent: screenshot failed")
        return []

    try:
        proposals: list[dict] = await _VisionModel().propose.remote.aio(
            screenshot_bytes, task, memory_context
        )
    except Exception:
        logger.exception("vision agent: Modal inference failed")
        return []

    actions: list[Action] = []
    for p in proposals:
        action = _parse_proposal(p)
        if action is not None:
            actions.append(action)

    return actions[:max_candidates]


def _parse_proposal(p: dict) -> Action | None:
    """Validate and convert one raw proposal dict into an Action."""
    try:
        raw_action = p.get("action", "")
        if raw_action not in _ACTION_TYPES:
            return None

        bbox_raw = p["bbox"]
        if len(bbox_raw) != 4:
            return None
        bbox: BBox = (
            float(bbox_raw[0]),
            float(bbox_raw[1]),
            float(bbox_raw[2]),
            float(bbox_raw[3]),
        )

        # Sanity check: bbox must have positive area
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None

        text = p.get("text") or None

        # type/select without text is a malformed proposal — reject it
        if raw_action in ("type", "select") and not text:
            return None

        return Action(
            action_type=ActionType(raw_action),
            bbox=bbox,
            source="vision",
            text=text,
            metadata={"raw": p},
        )
    except (KeyError, ValueError, TypeError):
        return None
