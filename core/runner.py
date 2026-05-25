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
import logging
from typing import Optional

from playwright.async_api import Page
import aiosqlite

logger = logging.getLogger(__name__)

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

MAX_STEPS = 25          # soft step limit — last resort only
LOOP_WINDOW = 2         # how many identical hashes trigger loop detection
EVALUATOR_INTERVAL = 5  # call evaluator every N successful steps


def _axtree_hash(axtree: str) -> str:
    """SHA-1 of the axtree text — used to detect unchanged page state."""
    return hashlib.sha1(axtree.encode()).hexdigest()



REPEAT_WINDOW = 3       # how many same-coordinate actions trigger coordinate loop

def _is_looping(session: Session, current_hash: str) -> bool:
    """
    Return True if the session is stuck in a loop.

    Two detection methods:
      1. Hash loop: last LOOP_WINDOW steps all failed with the same axtree
         hash — page is unchanged and nothing is working.
      2. Coordinate loop: last REPEAT_WINDOW steps all target the same bbox
         region — agent is clicking/typing the same element repeatedly,
         even if each action "succeeds" (e.g. toggling a checkbox).
    """
    recent = session.steps[-LOOP_WINDOW:]
    if (
        len(recent) == LOOP_WINDOW
        and all(s.axtree_hash == current_hash for s in recent)
        and all(not s.success for s in recent)
    ):
        return True

    recent_with_bbox = session.steps[-REPEAT_WINDOW:]
    if len(recent_with_bbox) == REPEAT_WINDOW and all(s.bbox for s in recent_with_bbox):
        first = recent_with_bbox[0]
        if all(
            s.action_type == first.action_type and _bbox_close(s.bbox, first.bbox)
            for s in recent_with_bbox[1:]
        ):
            return True

    return False


def _bbox_close(a: tuple, b: tuple, threshold: float = 20.0) -> bool:
    """True if two bboxes are within threshold pixels on all edges."""
    return all(abs(a[i] - b[i]) <= threshold for i in range(4))


async def run_session(
    page: Page,
    db: aiosqlite.Connection,
    session: Session,
    episode_metadata_base: dict | None = None,
    pending_rollbacks: dict[int, int] | None = None,
    agents: str = "both",
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
        logger.info(
            "step %d | url=%s | axtree_lines=%d | hash=%s",
            step_index, page.url, len(axtree_lines), current_hash[:8],
        )

        # ── 2. Retrieve memory context ─────────────────────────────────────────
        retrieved = memory.retrieve(session.task, axtree_snippet, top_k=3)
        memory_context = memory.format_for_prompt(retrieved) or None

        # ── 3. Run agent(s) ───────────────────────────────────────────────────
        dom_candidates: list = []
        vis_candidates: list = []
        dom_elapsed = 0.0
        vis_elapsed = 0.0

        if agents == "both":
            (dom_candidates, dom_elapsed), (vis_candidates, vis_elapsed) = (
                await asyncio.gather(
                    timed(dom_agent.propose_actions(page, session.task, memory_context=memory_context)),
                    timed(vision_agent.propose_actions(page, session.task, memory_context=memory_context)),
                )
            )
        elif agents == "dom":
            dom_candidates, dom_elapsed = await timed(
                dom_agent.propose_actions(page, session.task, memory_context=memory_context)
            )
        elif agents == "vision":
            vis_candidates, vis_elapsed = await timed(
                vision_agent.propose_actions(page, session.task, memory_context=memory_context)
            )

        episode_meta = {
            **(episode_metadata_base or {}),
            "dom_latency_ms":    round(dom_elapsed * 1000),
            "vision_latency_ms": round(vis_elapsed * 1000),
            "dom_candidates":    len(dom_candidates),
            "vision_candidates": len(vis_candidates),
            "dom_failed":        agents in ("both", "dom") and len(dom_candidates) == 0,
            "vision_failed":     agents in ("both", "vision") and len(vis_candidates) == 0,
            "step_index":        step_index,
            "agents":            agents,
        }

        all_candidates = dom_candidates + vis_candidates
        logger.info(
            "step %d | dom=%d (%.1fs) | vision=%d (%.1fs) | total=%d",
            step_index, len(dom_candidates), dom_elapsed,
            len(vis_candidates), vis_elapsed, len(all_candidates),
        )

        # ── 4. Agent done signal — nothing left to do ─────────────────────────
        if not all_candidates:
            verdict = await evaluate(session.task, session.steps, axtree_snippet)
            if verdict == "complete":
                termination_reason = "success"
            else:
                termination_reason = "premature_termination"
            break

        # ── 5. Score + rank ───────────────────────────────────────────────────
        scored = score_candidates(all_candidates, session.task)
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
        logger.info(
            "step %d | exec %s %s @ (%.0f,%.0f) score=%.3f src=%s",
            step_index, best.action.action_type.value,
            (best.action.text or "")[:30],
            best.action.center()[0], best.action.center()[1],
            best.score, best.action.source,
        )
        result = await execute(page, best)
        logger.info(
            "step %d | result success=%s signal=%s err=%s",
            step_index, result.success, result.signal, result.error,
        )
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
            bbox=best.action.bbox,
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

            # Structural heuristic A: submit detection via element name or task text.
            # A click that caused navigation + either the element is named
            # "submit/send/..." OR the task itself mentions submitting.
            submit_words = ("submit", "send", "confirm", "place", "order", "done")
            element_name = (best.action.metadata.get("name") or "").lower()
            task_lower = session.task.lower()
            element_match = any(w in element_name for w in submit_words)
            task_match = (
                best.action.action_type.value == "click"
                and any(w in task_lower for w in submit_words)
            )
            if result.success and (element_match or task_match):
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

    # ── Finalise session ──────────────────────────────────────────────────────
    final_reason = termination_reason or "step_limit"
    logger.info("session %d terminated: %s after %d steps", session.id, final_reason, len(session.steps))
    session.terminate(final_reason)
    await close_session(db, session.id, final_reason)

    return session


