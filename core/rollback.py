"""
Rollback logic for v3 multi-step sessions.

After each successful step, the runner saves a Checkpoint (url + axtree hash).
When a cascade is detected — or a manual rollback is requested via the API —
perform_rollback restores the browser and session state to that checkpoint.

Important limitation: rollback works for navigation-heavy tasks. It cannot
undo server-side side effects (form submissions, deletions, purchases).
This limitation is a research finding, not a bug — not all failure states
are recoverable. Rollback events are logged as labeled failure data.

Two rollback triggers:
    "cascade"  — automatic, triggered by N consecutive failed steps
    "manual"   — triggered by POST /session/{id}/rollback?to_step=N
"""

from playwright.async_api import Page
import aiosqlite

from core.session import Checkpoint, Session
from feedback.logger import log_rollback

# How many consecutive failures trigger an automatic cascade rollback
CASCADE_THRESHOLD = 3


def consecutive_failures(session: Session) -> int:
    """Count how many of the most recent steps failed in a row."""
    count = 0
    for step in reversed(session.steps):
        if not step.success:
            count += 1
        else:
            break
    return count


def should_cascade(session: Session) -> bool:
    """Return True if the session has hit the cascade threshold."""
    return (
        len(session.checkpoints) > 0  # nothing to rollback to otherwise
        and consecutive_failures(session) >= CASCADE_THRESHOLD
    )


async def perform_rollback(
    session: Session,
    page: Page,
    db: aiosqlite.Connection,
    to_step: int,
    reason: str = "cascade",
) -> Checkpoint | None:
    """
    Restore browser and session state to the checkpoint at to_step.

    Returns the Checkpoint we restored to, or None if no valid checkpoint
    exists at or before to_step (in which case nothing is changed).

    Steps:
        1. Find the most recent checkpoint at or before to_step
        2. Navigate browser to that checkpoint's URL
        3. Truncate session.steps and session.checkpoints
        4. Log the rollback event
    """
    # Find the best checkpoint at or before the requested step
    target = _find_checkpoint(session, to_step)
    if target is None:
        return None

    triggered_at = session.current_step_index

    # Navigate browser back to the checkpoint URL
    await page.goto(target.url, wait_until="domcontentloaded", timeout=15_000)

    # Truncate in-memory session state
    session.steps = [s for s in session.steps if s.step_index < target.step_index]
    session.checkpoints = [c for c in session.checkpoints if c.step_index <= target.step_index]

    # Log as labeled failure data
    await log_rollback(
        db,
        session_id=session.id,
        triggered_at_step=triggered_at,
        rolled_back_to_step=target.step_index,
        reason=reason,
    )

    return target


def _find_checkpoint(session: Session, to_step: int) -> Checkpoint | None:
    """
    Return the most recent checkpoint at or before to_step.
    Returns None if no such checkpoint exists.
    """
    candidates = [c for c in session.checkpoints if c.step_index <= to_step]
    return candidates[-1] if candidates else None
