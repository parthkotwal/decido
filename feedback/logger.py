import json
from typing import Optional

import asyncpg

from execution.executor import ExecutionResult


async def create_pool(dsn: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn)


async def log_execution(
    pool: asyncpg.Pool,
    task: str,
    result: ExecutionResult,
) -> int:
    """
    Persist one executed action and its outcome.

    Creates an episode row (task + page url) and an action_log row in a single
    transaction. Returns the action_log id — useful for correlating logs in tests.
    """
    action = result.action
    x1, y1, x2, y2 = action.bbox

    async with pool.acquire() as conn:
        async with conn.transaction():
            episode_id: int = await conn.fetchval(
                """
                INSERT INTO episodes (task, page_url)
                VALUES ($1, $2)
                RETURNING id
                """,
                task,
                result.url_before,
            )

            action_log_id: int = await conn.fetchval(
                """
                INSERT INTO action_logs (
                    episode_id,
                    source, action_type,
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    text, element_ref,
                    confidence, agreement, score,
                    success, signal,
                    url_before, url_after, mutation_count,
                    error, metadata
                ) VALUES (
                    $1,
                    $2, $3,
                    $4, $5, $6, $7,
                    $8, $9,
                    $10, $11, $12,
                    $13, $14,
                    $15, $16, $17,
                    $18, $19
                )
                RETURNING id
                """,
                episode_id,
                action.source, action.action_type.value,
                x1, y1, x2, y2,
                action.text, action.element_ref,
                action.confidence,
                result.agreement,
                result.score,
                result.success, result.signal,
                result.url_before, result.url_after, result.mutation_count,
                result.error, json.dumps(action.metadata),
            )

    return action_log_id
