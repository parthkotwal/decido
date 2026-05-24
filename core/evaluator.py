"""
Task completion evaluator.

A dedicated LM call — intentionally separate from the acting agents —
that reviews the full action history and current page state and returns
one of three verdicts:

    complete  — the task goal has been achieved
    incomplete — the task is not done yet; keep going
    stuck     — the agent is making no progress; should terminate

Separation of concerns: an agent deciding it's done is not the same as
an independent evaluation of whether the goal was achieved. The evaluator
uses the same model (gpt-5-nano) but a completely different prompt and role.

Triggered by:
    - Both agents returning no candidates (nothing left to do)
    - The same action being proposed twice in a row
    - A URL change that suggests navigation completed something
"""

import os
from openai import AsyncOpenAI
from core.session import StepRecord

MODEL = "gpt-5-nano"

_SYSTEM_PROMPT = """\
You are a task completion evaluator for a browser automation system. \
You are NOT the agent taking actions — your only job is to assess whether \
a given task has been completed based on the action history and current page state.

Rules:
  - If any step that was required to complete the task FAILED, return incomplete.
  - A page change or confirmation message alone is not sufficient — verify the \
required actions actually succeeded.
  - Return stuck only if the agent is clearly repeating the same action with no progress.

Respond with exactly one word:
  complete   — every required action succeeded and the goal has been achieved
  incomplete — the task is not yet done or a required step failed
  stuck      — the agent is repeating itself or making no progress

Do not explain. Output only one of those three words."""


def _build_prompt(task: str, history: list[StepRecord], axtree_snippet: str) -> str:
    """
    Summarise the session history and current page state for the evaluator.
    """
    lines = [f"Task: {task}", "", "Action history:"]
    if not history:
        lines.append("  (no actions taken yet)")
    else:
        for step in history:
            outcome = "SUCCESS" if step.success else "FAILED"
            action_detail = f"{step.action_type} via {step.action_source}"
            if step.element_name:
                action_detail += f' on "{step.element_name}"'
            if step.action_text:
                action_detail += f' with "{step.action_text}"'
            lines.append(
                f"  Step {step.step_index + 1}: {action_detail} → {outcome}"
            )

    lines += ["", "Current page (accessibility tree):"]
    lines.append(axtree_snippet[:2000] or "(empty)")

    return "\n".join(lines)


async def evaluate(
    task: str,
    history: list[StepRecord],
    axtree_snippet: str,
    api_key: str | None = None,
) -> str:
    """
    Call the evaluator LM and return 'complete', 'incomplete', or 'stuck'.

    Falls back to 'incomplete' on any API or parse error — safer to keep
    going than to terminate prematurely on a transient failure.
    """
    client = AsyncOpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])
    prompt = _build_prompt(task, history, axtree_snippet)

    try:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
            temperature=0.0,
        )
        verdict = (response.choices[0].message.content or "").strip().lower()
        if verdict in ("complete", "incomplete", "stuck"):
            return verdict
        return "incomplete"  # unknown response → keep going
    except Exception:
        return "incomplete"
