"""
Session runner — the observe → propose → rank → execute loop for v3.

The runner drives a Session to completion. It owns the termination logic
and coordinates all the pieces: agents, scorer, ranker, executor, memory,
evaluator, and logger.

Termination layers (cheapest first):
    1. Loop detection     — axtree hash unchanged 2+ steps, or same action twice → "loop"
    2. Soft step limit    — MAX_STEPS hit before any other signal → "step_limit"
    3. Agent done signal  — both agents return no candidates → trigger evaluator
    4. Evaluator verdict  — "complete" → "success" | "stuck" → "stuck" failure tag
    5. Post-execution     — URL change after action triggers evaluator as a check

The Session object (core/session.py) holds state.
The runner mutates it and persists each step via logger.py.
"""

import asyncio
import hashlib
from typing import Optional

from playwright.async_api import Page
import aiosqlite

from agents import dom_agent, vision_agent
from agents.dom_agent import _build_node_index
from core.evaluator import evaluate
from core.memory import EpisodicMemory
from core.ranker import select_best
from core.scorer import score_candidates
from core.session import Checkpoint, Session, StepRecord
from core.rollback import should_cascade, perform_rollback
from core.utils import timed
from execution.executor import execute
from feedback.logger import close_session, log_candidates, log_step

MAX_STEPS = 15          # soft step limit — last resort only
LOOP_WINDOW = 2         # how many identical hashes trigger loop detection
EVALUATOR_INTERVAL = 5  # call evaluator every N successful steps


def _axtree_hash(axtree: str) -> str:
    """SHA-1 of the axtree text — used to detect unchanged page state."""
    return hashlib.sha1(axtree.encode()).hexdigest()



def _is_looping(session: Session, current_hash: str) -> bool:
    """
    Return True if the session is stuck in a loop.

    Fires when the last LOOP_WINDOW steps all:
      - Have the same axtree hash as the current page (structure unchanged), AND
      - Failed (no success signal)

    Requiring both conditions prevents false positives on form-filling tasks
    where the agent correctly types into multiple fields on the same page
    (same axtree structure, but each step succeeds).
    """
    recent = session.steps[-LOOP_WINDOW:]
    return (
        len(recent) == LOOP_WINDOW
        and all(s.axtree_hash == current_hash for s in recent)
        and all(not s.success for s in recent)
    )


async def run_session(
    page: Page,
    db: aiosqlite.Connection,
    session: Session,
    episode_metadata_base: dict | None = None,
    pending_rollbacks: dict[int, int] | None = None,
) -> Session:
    """
    Drive session to completion and return the finished Session.

    page:              Playwright page, already navigated to session.start_url
    db:                open aiosqlite connection
    session:           Session object created by the caller (id already set from DB)
    episode_metadata_base: optional dict merged into each episode's metadata
    pending_rollbacks: shared dict {session_id: to_step} checked each iteration.
                       The rollback endpoint writes here; the runner consumes it.
    """
    memory = EpisodicMemory()
    termination_reason: Optional[str] = None
    pending_rollbacks = pending_rollbacks or {}

    while session.is_running:
        step_index = session.current_step_index

        # ── 0. Check for pending manual rollback request ──────────────────────
        if session.id in pending_rollbacks:
            to_step = pending_rollbacks.pop(session.id)
            await perform_rollback(session, page, db, to_step=to_step, reason="manual")
            continue

        # ── 1. Observe current page state ─────────────────────────────────────
        _, axtree_lines = await _build_node_index(page)
        axtree_snippet = "\n".join(axtree_lines)
        current_hash = _axtree_hash(axtree_snippet)

        # ── 2. Retrieve memory context ─────────────────────────────────────────
        retrieved = memory.retrieve(session.task, axtree_snippet, top_k=3)
        memory_context = memory.format_for_prompt(retrieved) or None

        # ── 3. Run both agents in parallel ────────────────────────────────────
        (dom_candidates, dom_elapsed), (vis_candidates, vis_elapsed) = (
            await asyncio.gather(
                timed(dom_agent.propose_actions(page, session.task, memory_context=memory_context)),
                timed(vision_agent.propose_actions(page, session.task, memory_context=memory_context)),
            )
        )

        episode_meta = {
            **(episode_metadata_base or {}),
            "dom_latency_ms":    round(dom_elapsed * 1000),
            "vision_latency_ms": round(vis_elapsed * 1000),
            "dom_candidates":    len(dom_candidates),
            "vision_candidates": len(vis_candidates),
            "dom_failed":        len(dom_candidates) == 0,
            "vision_failed":     len(vis_candidates) == 0,
            "step_index":        step_index,
        }

        all_candidates = dom_candidates + vis_candidates

        # ── 4. Agent done signal — nothing left to do ─────────────────────────
        if not all_candidates:
            verdict = await evaluate(session.task, session.steps, axtree_snippet)
            if verdict == "complete":
                termination_reason = "success"
            else:
                termination_reason = "premature_termination"
            break

        # ── 5. Score + rank ───────────────────────────────────────────────────
        scored = score_candidates(all_candidates)
        best = select_best(scored)

        if best is None:
            termination_reason = "action_failure"
            break

        # ── 6. Loop detection (before executing) ──────────────────────────────
        if _is_looping(session, current_hash):
            termination_reason = "loop"
            break

        # ── 7. Soft step limit ────────────────────────────────────────────────
        if step_index >= MAX_STEPS:
            termination_reason = "step_limit"
            break

        # ── 8. Execute ────────────────────────────────────────────────────────
        result = await execute(page, best)
        episode_id = await log_candidates(db, session.task, scored, best, result, episode_meta)

        # ── 9. Record step ────────────────────────────────────────────────────
        step_record = StepRecord(
            step_index=step_index,
            episode_id=episode_id,
            page_url=result.url_before,
            axtree_hash=current_hash,
            action_type=best.action.action_type.value,
            action_source=best.action.source,
            success=result.success,
            action_text=best.action.text,
            element_name=best.action.metadata.get("name"),
        )

        is_checkpoint = result.success
        session.add_step(step_record)
        await log_step(db, session.id, step_record, checkpoint=is_checkpoint)

        if result.success:
            session.add_checkpoint(Checkpoint(
                step_index=step_index,
                url=result.url_after,
                axtree_hash=_axtree_hash(axtree_snippet),
            ))

        # ── 10. Store in memory ───────────────────────────────────────────────
        memory.store(step_record, axtree_snippet)

        # ── 11. Cascade detection — rollback if too many consecutive failures ──
        if should_cascade(session):
            checkpoint = await perform_rollback(
                session, page, db,
                to_step=session.current_step_index,
                reason="cascade",
            )
            if checkpoint is None:
                # No checkpoint to roll back to — terminate instead
                termination_reason = "cascade"
                break
            # Rolled back successfully — continue the loop from checkpoint
            continue

        # ── 13. Post-execution evaluator check ────────────────────────────────
        # Trigger 1: URL change — re-extract axtree from the new page first
        if result.url_after and result.url_after != result.url_before:
            _, new_axtree_lines = await _build_node_index(page)
            new_axtree = "\n".join(new_axtree_lines)

            # Structural heuristic A: explicit submit element name + URL change.
            # A click on something named "submit/send/confirm/..." that caused
            # navigation is a reliable completion signal without needing the LM.
            element_name = (best.action.metadata.get("name") or "").lower()
            submit_words = ("submit", "send", "confirm", "place", "order", "done")
            if result.success and any(w in element_name for w in submit_words):
                termination_reason = "success"
                break

            # Structural heuristic B: URL changed, new page has no interactive
            # elements, and recent steps succeeded. The agent completed something
            # and landed on a terminal page — strong completion signal regardless
            # of task. Still call the evaluator, but pass a flag in the axtree
            # so it has this context.
            recent_successes = sum(1 for s in session.steps[-3:] if s.success)
            if not new_axtree_lines and recent_successes >= 2:
                verdict = await evaluate(session.task, session.steps, "(no interactive elements on new page)")
                if verdict in ("complete", "stuck"):
                    termination_reason = "success" if verdict == "complete" else "loop"
                    break

            # General case: let the evaluator judge with the full new axtree
            verdict = await evaluate(session.task, session.steps, new_axtree)
            if verdict == "complete":
                termination_reason = "success"
                break
            elif verdict == "stuck":
                termination_reason = "loop"
                break

        # Trigger 2: periodic check every N successful steps — catches cases
        # where the agent is filling a form but never navigates (no URL change)
        successful_steps = sum(1 for s in session.steps if s.success)
        if successful_steps > 0 and successful_steps % EVALUATOR_INTERVAL == 0:
            verdict = await evaluate(session.task, session.steps, axtree_snippet)
            if verdict == "complete":
                termination_reason = "success"
                break
            elif verdict == "stuck":
                termination_reason = "loop"
                break

        # ── 14. Error on execution → action_failure ───────────────────────────
        if result.error:
            termination_reason = "action_failure"
            break

    # ── Finalise session ──────────────────────────────────────────────────────
    final_reason = termination_reason or "step_limit"
    session.terminate(final_reason)
    await close_session(db, session.id, final_reason)

    return session


