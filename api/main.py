import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
from typing import Optional

import aiosqlite
from fastapi import FastAPI, HTTPException
from playwright.async_api import async_playwright, Browser
from pydantic import BaseModel

from agents import dom_agent, vision_agent
from core.ranker import select_best
from core.rollback import perform_rollback
from core.runner import run_session
from core.scorer import score_candidates
from core.session import Session
from core.utils import timed
from execution.executor import execute
from feedback.logger import DB_PATH, init_db, log_candidates, create_session


# ── shared state ──────────────────────────────────────────────────────────────

class AppState:
    browser: Browser
    db: aiosqlite.Connection
    # Sessions currently running, keyed by session id
    active_sessions: dict[int, Session]
    # Rollback requests from the endpoint, consumed by the runner each iteration
    # {session_id: to_step}
    pending_rollbacks: dict[int, int]


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pw = await async_playwright().start()
    state.browser = await pw.chromium.launch(headless=True)
    state.db = await aiosqlite.connect(DB_PATH)
    state.active_sessions = {}
    state.pending_rollbacks = {}
    await init_db(state.db)
    yield
    await state.browser.close()
    await pw.stop()
    await state.db.close()


app = FastAPI(title="Decido", lifespan=lifespan)


# ── request / response models ─────────────────────────────────────────────────

class ActRequest(BaseModel):
    task: str
    url: str


class TaskRequest(BaseModel):
    task: str
    url: str


class ActionDetail(BaseModel):
    source: str
    action_type: str
    bbox: tuple[float, float, float, float]
    text: Optional[str]
    agreement: float
    score: float


class ActResponse(BaseModel):
    success: bool
    signal: Optional[str]
    url_before: str
    url_after: str
    action: ActionDetail
    error: Optional[str] = None


class StepSummary(BaseModel):
    step_index: int
    action_type: str
    action_source: str
    success: bool
    page_url: str


class TaskResponse(BaseModel):
    session_id: int
    status: str                     # "complete" | "failed"
    termination_reason: str
    step_count: int
    steps: list[StepSummary]


class RollbackResponse(BaseModel):
    session_id: int
    rolled_back_to_step: int
    current_url: str
    message: str


# ── v2: single-step endpoint (kept for testing) ───────────────────────────────

@app.post("/act", response_model=ActResponse)
async def act(req: ActRequest) -> ActResponse:
    page = await state.browser.new_page()

    try:
        await page.goto(req.url, wait_until="domcontentloaded", timeout=15_000)

        (dom_candidates, dom_elapsed), (vision_candidates, vision_elapsed) = (
            await asyncio.gather(
                timed(dom_agent.propose_actions(page, req.task)),
                timed(vision_agent.propose_actions(page, req.task)),
            )
        )

        episode_metadata = {
            "dom_latency_ms":    round(dom_elapsed * 1000),
            "vision_latency_ms": round(vision_elapsed * 1000),
            "dom_candidates":    len(dom_candidates),
            "vision_candidates": len(vision_candidates),
            "dom_failed":        len(dom_candidates) == 0,
            "vision_failed":     len(vision_candidates) == 0,
        }

        all_candidates = dom_candidates + vision_candidates
        if not all_candidates:
            raise HTTPException(status_code=422, detail="No action candidates found")

        scored = score_candidates(all_candidates, req.task)
        best = select_best(scored)

        if best is None:
            raise HTTPException(status_code=422, detail="Ranking produced no result")

        result = await execute(page, best)
        await log_candidates(state.db, req.task, scored, best, result, episode_metadata)

        return ActResponse(
            success=result.success,
            signal=result.signal,
            url_before=result.url_before,
            url_after=result.url_after,
            action=ActionDetail(
                source=best.action.source,
                action_type=best.action.action_type.value,
                bbox=best.action.bbox,
                text=best.action.text,
                agreement=best.agreement,
                score=best.score,
            ),
            error=result.error,
        )

    finally:
        await page.close()


# ── v3: multi-step session endpoint ──────────────────────────────────────────

@app.post("/task", response_model=TaskResponse)
async def task(req: TaskRequest) -> TaskResponse:
    """
    Run a full multi-step session until the task is complete or terminated.

    Opens a browser page, loops (observe → propose → rank → execute) with
    memory and termination logic, and returns a summary when done.

    This request blocks until the session ends. To request a rollback
    mid-session, POST /session/{id}/rollback from another client while
    this request is in flight.
    """
    page = await state.browser.new_page()

    try:
        await page.goto(req.url, wait_until="domcontentloaded", timeout=15_000)

        session_id = await create_session(state.db, req.task, req.url)
        session = Session(id=session_id, task=req.task, start_url=req.url)
        state.active_sessions[session_id] = session

        try:
            await run_session(
                page=page,
                db=state.db,
                session=session,
                pending_rollbacks=state.pending_rollbacks,
            )
        finally:
            state.active_sessions.pop(session_id, None)

        return TaskResponse(
            session_id=session.id,
            status="complete" if session.termination_reason == "success" else "failed",
            termination_reason=session.termination_reason or "step_limit",
            step_count=len(session.steps),
            steps=[
                StepSummary(
                    step_index=s.step_index,
                    action_type=s.action_type,
                    action_source=s.action_source,
                    success=s.success,
                    page_url=s.page_url,
                )
                for s in session.steps
            ],
        )

    finally:
        await page.close()


# ── v3: rollback endpoint ─────────────────────────────────────────────────────

@app.post("/session/{session_id}/rollback", response_model=RollbackResponse)
async def rollback(session_id: int, to_step: int) -> RollbackResponse:
    """
    Request a rollback for a running or completed session.

    For running sessions: queues the rollback to be processed on the next
    loop iteration of the runner.

    For completed sessions: performs the rollback immediately (navigates
    browser to checkpoint URL and updates session state in DB).
    """
    # ── Active session: queue for the runner to pick up ───────────────────────
    if session_id in state.active_sessions:
        session = state.active_sessions[session_id]
        checkpoint = next(
            (c for c in reversed(session.checkpoints) if c.step_index <= to_step),
            None,
        )
        if checkpoint is None:
            raise HTTPException(
                status_code=400,
                detail=f"No checkpoint at or before step {to_step}",
            )
        state.pending_rollbacks[session_id] = to_step
        return RollbackResponse(
            session_id=session_id,
            rolled_back_to_step=checkpoint.step_index,
            current_url=checkpoint.url,
            message="Rollback queued — will apply on next loop iteration.",
        )

    # ── Completed session: perform immediately ─────────────────────────────────
    # Look up session from DB to verify it exists
    row = await state.db.execute_fetchall(
        "SELECT id, status FROM sessions WHERE id = ?", (session_id,)
    )
    if not row:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    # Reconstruct minimal session state from DB for rollback
    steps_rows = await state.db.execute_fetchall(
        "SELECT step_index, page_url, axtree_hash, checkpoint FROM steps "
        "WHERE session_id = ? ORDER BY step_index",
        (session_id,),
    )
    from core.session import Checkpoint as CP, StepRecord as SR
    reconstructed = Session(id=session_id, task="", start_url="")
    for r in steps_rows:
        reconstructed.steps.append(
            SR(r[0], 0, r[1], r[2], "", "", False)
        )
        if r[3]:  # checkpoint flag
            reconstructed.checkpoints.append(CP(r[0], r[1], r[2]))

    page = await state.browser.new_page()
    try:
        checkpoint = await perform_rollback(
            reconstructed, page, state.db, to_step=to_step, reason="manual"
        )
        if checkpoint is None:
            raise HTTPException(
                status_code=400,
                detail=f"No checkpoint at or before step {to_step}",
            )
        return RollbackResponse(
            session_id=session_id,
            rolled_back_to_step=checkpoint.step_index,
            current_url=page.url,
            message="Rollback complete.",
        )
    finally:
        await page.close()
