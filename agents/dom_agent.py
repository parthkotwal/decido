import asyncio
from typing import Optional

from playwright.async_api import Page

from core.action import Action, ActionType, BBox

# Accessibility roles we consider interactive and their default action type
_ROLE_TO_ACTION: dict[str, ActionType] = {
    "button": ActionType.CLICK,
    "link": ActionType.CLICK,
    "menuitem": ActionType.CLICK,
    "option": ActionType.CLICK,
    "checkbox": ActionType.CLICK,
    "radio": ActionType.CLICK,
    "tab": ActionType.CLICK,
    "textbox": ActionType.TYPE,
    "searchbox": ActionType.TYPE,
    "combobox": ActionType.TYPE,
    "spinbutton": ActionType.TYPE,
    "listbox": ActionType.SELECT,
}


def _flatten_axtree(node: dict, results: list[dict]) -> None:
    """Walk the axtree depth-first, collect interactive leaf-ish nodes."""
    role = node.get("role", "")
    name = node.get("name", "")

    if role in _ROLE_TO_ACTION and name:
        results.append(node)

    for child in node.get("children", []):
        _flatten_axtree(child, results)


def _task_relevance(node_name: str, task: str) -> float:
    """Rough heuristic: fraction of task words present in the element name."""
    task_words = set(task.lower().split())
    name_words = set(node_name.lower().split())
    if not task_words:
        return 0.0
    return len(task_words & name_words) / len(task_words)


async def _resolve_bbox(page: Page, role: str, name: str) -> Optional[BBox]:
    """Locate an element by role+name and return its bounding box."""
    try:
        locator = page.get_by_role(role, name=name)  # type: ignore[arg-type]
        # Use first() in case multiple elements match — first visible is safest
        box = await locator.first.bounding_box(timeout=2000)
        if box is None:
            return None
        return (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"])
    except Exception:
        return None


async def propose_actions(
    page: Page,
    task: str,
    max_candidates: int = 3,
) -> list[Action]:
    """
    Snapshot the axtree, resolve bboxes, and return ranked candidate Actions.

    Candidates are sorted by confidence descending. Elements whose bbox cannot
    be resolved (hidden, off-screen, stale) are silently dropped.
    """
    snapshot = await page.accessibility.snapshot()
    if snapshot is None:
        return []

    interactive_nodes: list[dict] = []
    _flatten_axtree(snapshot, interactive_nodes)

    # Resolve bboxes concurrently — one coroutine per candidate
    async def build_action(node: dict) -> Optional[Action]:
        role = node["role"]
        name = node.get("name", "")
        action_type = _ROLE_TO_ACTION[role]

        bbox = await _resolve_bbox(page, role, name)
        if bbox is None:
            return None

        relevance = _task_relevance(name, task)
        focused_bonus = 0.2 if node.get("focused") else 0.0
        confidence = min(1.0, 0.4 + 0.4 * relevance + focused_bonus)

        return Action(
            action_type=action_type,
            bbox=bbox,
            source="dom",
            confidence=confidence,
            element_ref=node.get("nodeId"),
            metadata={"role": role, "name": name},
        )

    results = await asyncio.gather(*[build_action(n) for n in interactive_nodes])

    candidates = [a for a in results if a is not None]
    candidates.sort(key=lambda a: a.confidence, reverse=True)
    return candidates[:max_candidates]
