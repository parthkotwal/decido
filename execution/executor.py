import asyncio
from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import Page

from core.action import Action, ActionType
from core.scorer import ScoredAction

SETTLE_DELAY = 0.5       # seconds to wait after action before checking success
MUTATION_THRESHOLD = 3   # minimum DOM mutations to count as a meaningful change

# JS injected before execution — installs a MutationObserver on the full document
_OBSERVER_INSTALL = """
() => {
    window.__decidoMutations = 0;
    window.__decidoObserver = new MutationObserver((records) => {
        window.__decidoMutations += records.length;
    });
    window.__decidoObserver.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        characterData: true,
    });
}
"""

_OBSERVER_READ = """
() => {
    const count = window.__decidoMutations || 0;
    if (window.__decidoObserver) window.__decidoObserver.disconnect();
    return count;
}
"""


@dataclass
class ExecutionResult:
    action: Action
    score: float
    agreement: float
    success: bool
    signal: Optional[str] = None   # "url_change" | "dom_mutation" | "input_value" | None
    url_before: str = ""
    url_after: str = ""
    mutation_count: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


async def execute(page: Page, scored: ScoredAction) -> ExecutionResult:
    """
    Execute the selected action on the page and return an ExecutionResult.

    Success is determined by observing side effects after execution:
      1. URL change  — navigation occurred
      2. DOM mutation count >= MUTATION_THRESHOLD  — page content changed
    """
    action = scored.action
    result = ExecutionResult(
        action=action,
        score=scored.score,
        agreement=scored.agreement,
        success=False,
        url_before=page.url,
    )

    try:
        url_before = page.url
        await page.evaluate(_OBSERVER_INSTALL)

        await _dispatch(page, action)
        await asyncio.sleep(SETTLE_DELAY)

        # Capture URL first — if the page navigated, the JS context for
        # the mutation observer is gone and evaluate() would throw.
        url_after = page.url
        result.url_after = url_after

        if url_after != url_before:
            result.success = True
            result.signal = "url_change"
        else:
            # Page didn't navigate — safe to read the observer
            try:
                mutation_count = await page.evaluate(_OBSERVER_READ)
            except Exception:
                mutation_count = 0
            result.mutation_count = mutation_count

            if mutation_count >= MUTATION_THRESHOLD:
                result.success = True
                result.signal = "dom_mutation"
            elif action.action_type in (ActionType.TYPE, ActionType.SELECT):
                typed = action.text or ""
                if typed and await _input_has_value(page, action.center(), typed):
                    result.success = True
                    result.signal = "input_value"
            elif action.action_type == ActionType.CLICK:
                if await _checkbox_is_checked(page, action.center()):
                    result.success = True
                    result.signal = "checkbox_checked"
                elif await _clicked_focusable(page, action.center()):
                    result.success = True
                    result.signal = "focus_change"

    except Exception as e:
        result.url_after = page.url
        result.error = str(e)

    return result


async def _dispatch(page: Page, action: Action) -> None:
    """Route to the correct Playwright call based on action type."""
    x, y = action.center()

    if action.action_type == ActionType.CLICK:
        await page.mouse.click(x, y)

    elif action.action_type == ActionType.TYPE:
        await page.mouse.click(x, y)          # focus first
        await page.keyboard.type(action.text or "")

    elif action.action_type == ActionType.HOVER:
        await page.mouse.move(x, y)

    elif action.action_type == ActionType.SCROLL:
        direction = action.scroll_direction or "down"
        amount = action.scroll_amount or 300
        delta_y = amount if direction == "down" else -amount
        await page.mouse.wheel(0, delta_y)

    elif action.action_type == ActionType.SELECT:
        # <select> elements respond to page.select_option; locate by bbox center
        elements = await page.query_selector_all(
            f"*:is(select)"
        )
        for el in elements:
            box = await el.bounding_box()
            if box and _point_in_box(x, y, box):
                await el.select_option(value=action.text or "")
                return
        # fallback: click the center and hope it's a custom dropdown
        await page.mouse.click(x, y)


async def _input_has_value(page: Page, center: tuple[float, float], expected: str) -> bool:
    """Check whether the input/textarea at (x, y) now contains the expected text."""
    x, y = center
    try:
        value: str = await page.evaluate(
            """([x, y]) => {
                let el = document.elementFromPoint(x, y);
                if (!el) return '';
                if (el.value !== undefined && el.value !== '') return el.value;

                // Click may have landed on a label — find the associated input
                const label = el.closest('label');
                if (label) {
                    const input = label.querySelector('input, textarea');
                    if (input && input.value) return input.value;
                    if (label.htmlFor) {
                        const ref = document.getElementById(label.htmlFor);
                        if (ref && ref.value) return ref.value;
                    }
                }

                return el.textContent ?? '';
            }""",
            [x, y],
        )
        return expected in value
    except Exception:
        return False


async def _checkbox_is_checked(page: Page, center: tuple[float, float]) -> bool:
    """Return True if the element at (x, y) is a checked checkbox or radio button.

    Handles the common case where the click lands on a <label> wrapping the
    input — walks up to the parent label and checks its associated input.
    """
    x, y = center
    try:
        return await page.evaluate(
            """([x, y]) => {
                let el = document.elementFromPoint(x, y);
                if (!el) return false;

                // Direct hit on the input itself
                if (el.tagName === 'INPUT') {
                    const t = (el.type || '').toLowerCase();
                    if ((t === 'checkbox' || t === 'radio') && el.checked) return true;
                }

                // Click landed on a label — find the associated input
                const label = el.closest('label');
                if (label) {
                    const input = label.querySelector('input[type="checkbox"], input[type="radio"]');
                    if (input && input.checked) return true;
                    if (label.htmlFor) {
                        const ref = document.getElementById(label.htmlFor);
                        if (ref && ref.checked) return true;
                    }
                }

                return false;
            }""",
            [x, y],
        )
    except Exception:
        return False


async def _clicked_focusable(page: Page, center: tuple[float, float]) -> bool:
    """Return True if a text-entry element at (x, y) is now the active element.

    Only counts text inputs/textareas — not checkboxes, radios, or buttons,
    which have their own success signals.
    """
    x, y = center
    try:
        return await page.evaluate(
            """([x, y]) => {
                let el = document.elementFromPoint(x, y);
                if (!el) return false;

                // Walk up from labels
                const label = el.closest('label');
                if (label) {
                    const input = label.querySelector('input, textarea, select');
                    if (input) el = input;
                    else if (label.htmlFor) {
                        const ref = document.getElementById(label.htmlFor);
                        if (ref) el = ref;
                    }
                }

                const target = el.closest('input, textarea, select, [contenteditable]');
                if (!target) return false;

                // Exclude checkboxes/radios — they have their own signal
                const tag = target.tagName.toLowerCase();
                if (tag === 'input') {
                    const t = (target.type || '').toLowerCase();
                    if (t === 'checkbox' || t === 'radio' || t === 'submit' || t === 'button') return false;
                }

                return document.activeElement === target;
            }""",
            [x, y],
        )
    except Exception:
        return False


def _point_in_box(x: float, y: float, box: dict) -> bool:
    return (
        box["x"] <= x <= box["x"] + box["width"]
        and box["y"] <= y <= box["y"] + box["height"]
    )
