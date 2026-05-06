import asyncio
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()
from typing import Optional

import aiosqlite
from fastapi import FastAPI, HTTPException
from playwright.async_api import async_playwright, Browser
from pydantic import BaseModel

from agents import dom_agent, vision_agent
from core.scorer import score_candidates
from core.ranker import select_best
from execution.executor import execute
from feedback.logger import DB_PATH, init_db, log_candidates


# ── shared state ──────────────────────────────────────────────────────────────

class AppState:
    browser: Browser
    db: aiosqlite.Connection


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pw = await async_playwright().start()
    state.browser = await pw.chromium.launch(headless=True)
    state.db = await aiosqlite.connect(DB_PATH)
    await init_db(state.db)
    yield
    await state.browser.close()
    await pw.stop()
    await state.db.close()


app = FastAPI(title="Decido", lifespan=lifespan)


# ── schema ────────────────────────────────────────────────────────────────────

class ActRequest(BaseModel):
    task: str
    url: str


class ActionDetail(BaseModel):
    source: str
    action_type: str
    bbox: tuple[float, float, float, float]
    text: Optional[str]
    confidence: float
    agreement: float
    score: float


class ActResponse(BaseModel):
    success: bool
    signal: Optional[str]
    url_before: str
    url_after: str
    action: ActionDetail
    error: Optional[str] = None


# ── endpoint ──────────────────────────────────────────────────────────────────

@app.post("/act", response_model=ActResponse)
async def act(req: ActRequest) -> ActResponse:
    page = await state.browser.new_page()

    try:
        await page.goto(req.url, wait_until="domcontentloaded", timeout=15_000)

        # Run both agents in parallel, capturing latency and failure status
        (dom_candidates, dom_elapsed), (vision_candidates, vision_elapsed) = (
            await asyncio.gather(
                _timed(dom_agent.propose_actions(page, req.task)),
                _timed(vision_agent.propose_actions(page, req.task)),
            )
        )

        episode_metadata = {
            "dom_latency_ms":       round(dom_elapsed * 1000),
            "vision_latency_ms":    round(vision_elapsed * 1000),
            "dom_candidates":       len(dom_candidates),
            "vision_candidates":    len(vision_candidates),
            "dom_failed":           len(dom_candidates) == 0,
            "vision_failed":        len(vision_candidates) == 0,
        }

        all_candidates = dom_candidates + vision_candidates
        if not all_candidates:
            raise HTTPException(status_code=422, detail="No action candidates found")

        scored = score_candidates(all_candidates)
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
                confidence=best.confidence,
                agreement=best.agreement,
                score=best.score,
            ),
            error=result.error,
        )

    finally:
        await page.close()


async def _timed(coro) -> tuple[list, float]:
    """Run a coroutine and return (result, elapsed_seconds)."""
    t0 = time.monotonic()
    result = await coro
    return result, time.monotonic() - t0
