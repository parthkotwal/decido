import asyncio
import json
import os
from typing import Optional

from google import genai
from google.genai import types
from playwright.async_api import Page

from core.action import Action, ActionType, BBox

GEMINI_MODEL = "gemini-2.5-flash-lite"

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

_SYSTEM_PROMPT = """\
You are a browser automation agent. Given an accessibility tree and a task, \
propose up to 3 actions that would best complete the task.

Respond ONLY with a JSON array. Each element must have:
  - node_id   (integer from the tree)
  - action    (one of: click, type, scroll, select, hover)
  - confidence (float 0–1)
  - text      (string, required for "type" and "select", omit otherwise)

Example:
[
  {"node_id": 12, "action": "click", "confidence": 0.92},
  {"node_id": 7,  "action": "type",  "confidence": 0.80, "text": "hello@example.com"}
]
"""


# ── axtree helpers ────────────────────────────────────────────────────────────

async def _resolve_bbox(page: Page, role: str, name: str) -> Optional[BBox]:
    try:
        box = await page.get_by_role(role, name=name).first.bounding_box(timeout=2000)  # type: ignore[arg-type]
        if box is None:
            return None
        return (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"])
    except Exception:
        return None


async def _build_node_index(
    page: Page, snapshot: dict
) -> tuple[dict[int, dict], list[str]]:
    """
    Walk the axtree, resolve bboxes, return:
      - index: {node_id -> {role, name, action_type, bbox, element_ref}}
      - lines: text lines for the Gemini prompt
    """
    counter = [0]
    index: dict[int, dict] = {}
    lines: list[str] = []
    resolve_tasks: list[tuple[int, str, str]] = []  # (node_id, role, name)

    def walk(node: dict) -> None:
        role = node.get("role", "")
        name = node.get("name", "")
        if role in _ROLE_TO_ACTION and name:
            nid = counter[0]
            counter[0] += 1
            index[nid] = {
                "role": role,
                "name": name,
                "action_type": _ROLE_TO_ACTION[role],
                "element_ref": node.get("nodeId"),
                "bbox": None,
            }
            resolve_tasks.append((nid, role, name))
        for child in node.get("children", []):
            walk(child)

    walk(snapshot)

    # Resolve all bboxes concurrently
    async def resolve(nid: int, role: str, name: str) -> None:
        index[nid]["bbox"] = await _resolve_bbox(page, role, name)

    await asyncio.gather(*[resolve(nid, r, n) for nid, r, n in resolve_tasks])

    # Build prompt lines (only nodes with a resolved bbox)
    for nid, info in index.items():
        if info["bbox"] is None:
            continue
        x1, y1, x2, y2 = (round(v) for v in info["bbox"])
        lines.append(
            f'[{nid}] {info["role"]} "{info["name"]}" bbox=({x1},{y1},{x2},{y2})'
        )

    return index, lines


# ── Gemini call ───────────────────────────────────────────────────────────────

def _build_prompt(task: str, axtree_lines: list[str]) -> str:
    tree_text = "\n".join(axtree_lines) if axtree_lines else "(no interactive elements found)"
    return f"Task: {task}\n\nAccessibility tree:\n{tree_text}"


def _parse_gemini_response(text: str) -> list[dict]:
    """Extract the JSON array from Gemini's response, tolerating markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


# ── public API ────────────────────────────────────────────────────────────────

async def propose_actions(
    page: Page,
    task: str,
    max_candidates: int = 3,
    api_key: Optional[str] = None,
) -> list[Action]:
    """
    Use Gemini 2.5 Flash Lite to propose candidate Actions from the page axtree.

    Returns up to max_candidates Actions sorted by confidence descending.
    Returns an empty list (rather than raising) on any API or parse failure.
    """
    snapshot = await page.accessibility.snapshot()
    if snapshot is None:
        return []

    index, axtree_lines = await _build_node_index(page, snapshot)
    if not axtree_lines:
        return []

    client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
    prompt = _build_prompt(task, axtree_lines)

    try:
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.0,
            ),
        )
        raw = response.text or ""
    except Exception:
        return []

    proposals = _parse_gemini_response(raw)

    actions: list[Action] = []
    for p in proposals:
        try:
            nid = int(p["node_id"])
            node = index.get(nid)
            if node is None or node["bbox"] is None:
                continue

            action_type = ActionType(p["action"])
            confidence = float(p.get("confidence", 0.5))
            text = p.get("text")

            actions.append(
                Action(
                    action_type=action_type,
                    bbox=node["bbox"],
                    source="dom",
                    confidence=min(1.0, max(0.0, confidence)),
                    text=text,
                    element_ref=node["element_ref"],
                    metadata={"role": node["role"], "name": node["name"]},
                )
            )
        except (KeyError, ValueError):
            continue

    actions.sort(key=lambda a: a.confidence, reverse=True)
    return actions[:max_candidates]
