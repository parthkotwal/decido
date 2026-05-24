import json
import os

from openai import AsyncOpenAI
from playwright.async_api import Page

from core.action import Action, ActionType, BBox

MODEL = "gpt-5-nano"

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
  - text      (string, required for "type" and "select", omit otherwise)

Example:
[
  {"node_id": 12, "action": "click"},
  {"node_id": 7,  "action": "type", "text": "hello@example.com"}
]
"""


# ── DOM extraction ────────────────────────────────────────────────────────────

_EXTRACT_JS = """
() => {
    const SELECTORS = [
        'button', 'a[href]', 'input', 'select', 'textarea',
        '[role="button"]', '[role="link"]', '[role="checkbox"]',
        '[role="radio"]', '[role="textbox"]', '[role="combobox"]',
        '[role="menuitem"]', '[role="tab"]', '[role="option"]',
        '[role="searchbox"]', '[role="spinbutton"]', '[role="listbox"]',
    ].join(',');

    const TAG_TO_ROLE = {
        button: 'button', a: 'link', select: 'listbox',
        textarea: 'textbox',
    };
    const INPUT_TYPE_TO_ROLE = {
        checkbox: 'checkbox', radio: 'radio', text: 'textbox',
        email: 'textbox', password: 'textbox', search: 'searchbox',
        number: 'spinbutton',
    };

    function resolveLabel(el) {
        if (el.getAttribute('aria-label')) return el.getAttribute('aria-label').trim();

        const labelledBy = el.getAttribute('aria-labelledby');
        if (labelledBy) {
            const ref = document.getElementById(labelledBy);
            if (ref) return ref.textContent.trim();
        }

        if (el.id) {
            const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
            if (label) return label.textContent.trim().slice(0, 80);
        }

        const parentLabel = el.closest('label');
        if (parentLabel) return parentLabel.textContent.trim().slice(0, 80);

        return (
            el.getAttribute('placeholder')
            || el.getAttribute('title')
            || el.innerText?.trim().slice(0, 80)
            || el.getAttribute('value')
            || ''
        ).trim();
    }

    return Array.from(document.querySelectorAll(SELECTORS))
        .map((el, i) => {
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return null;
            if (rect.bottom < 0 || rect.top > window.innerHeight) return null;

            const tag = el.tagName.toLowerCase();
            const explicitRole = el.getAttribute('role');
            let role = explicitRole
                || (tag === 'input' ? INPUT_TYPE_TO_ROLE[el.type] || 'textbox' : null)
                || TAG_TO_ROLE[tag]
                || tag;

            // Capture current value for inputs/textareas so the agent
            // can see which fields are already filled
            const value = (tag === 'input' || tag === 'textarea')
                ? (el.value || '')
                : '';

            return {
                index: i,
                role,
                name: resolveLabel(el),
                bbox: [rect.left, rect.top, rect.right, rect.bottom],
                focused: document.activeElement === el,
                value,
            };
        })
        .filter(Boolean);
}
"""


async def _build_node_index(page: Page) -> tuple[dict[int, dict], list[str]]:
    elements: list[dict] = await page.evaluate(_EXTRACT_JS)

    index: dict[int, dict] = {}
    lines: list[str] = []

    for nid, el in enumerate(elements):
        role = el["role"]
        action_type = _ROLE_TO_ACTION.get(role)
        if action_type is None:
            continue

        x1, y1, x2, y2 = el["bbox"]
        bbox: BBox = (x1, y1, x2, y2)
        name = el["name"] or role

        value = el.get("value", "")
        index[nid] = {
            "role": role,
            "name": name,
            "action_type": action_type,
            "bbox": bbox,
            "focused": el["focused"],
            "value": value,
        }
        value_str = f' value="{value}"' if value else ""
        lines.append(
            f'[{nid}] {role} "{name}"{value_str} bbox=({round(x1)},{round(y1)},{round(x2)},{round(y2)})'
        )

    return index, lines


# ── LM call ───────────────────────────────────────────────────────────────────

def _build_prompt(task: str, axtree_lines: list[str], memory_context: str | None = None) -> str:
    tree_text = "\n".join(axtree_lines) if axtree_lines else "(no interactive elements found)"
    prompt = f"Task: {task}\n\nAccessibility tree:\n{tree_text}"
    if memory_context:
        prompt += f"\n\n{memory_context}"
    return prompt


def _parse_response(text: str) -> list[dict]:
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
    api_key: str | None = None,
    memory_context: str | None = None,
) -> list[Action]:
    """
    Use an LM to propose candidate Actions from the page DOM.

    memory_context: optional retrieved history block from EpisodicMemory,
    injected below the axtree so the agent knows what has already been tried.

    Returns up to max_candidates Actions.
    Returns an empty list (rather than raising) on any API or parse failure.
    """
    index, axtree_lines = await _build_node_index(page)
    if not axtree_lines:
        return []

    client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    prompt = _build_prompt(task, axtree_lines, memory_context)

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or ""
    except Exception:
        return []

    proposals = _parse_response(raw)

    actions: list[Action] = []
    for p in proposals:
        try:
            nid = int(p["node_id"])
            node = index.get(nid)
            if node is None or node["bbox"] is None:
                continue

            text = p.get("text") or None
            if p["action"] in ("type", "select") and not text:
                continue

            actions.append(
                Action(
                    action_type=ActionType(p["action"]),
                    bbox=node["bbox"],
                    source="dom",
                    text=text,
                    element_ref=node.get("element_ref"),
                    metadata={"role": node["role"], "name": node["name"]},
                )
            )
        except (KeyError, ValueError):
            continue

    return actions[:max_candidates]
