import json
import aiosqlite

from execution.executor import ExecutionResult

DB_PATH = "decido.db"


async def init_db(db: aiosqlite.Connection) -> None:
    with open("db/schema.sql") as f:
        await db.executescript(f.read())
    await db.commit()


async def log_execution(db: aiosqlite.Connection, task: str, result: ExecutionResult) -> int:
    """
    Persist one executed action and its outcome.

    Returns the action_log id.
    """
    action = result.action
    x1, y1, x2, y2 = action.bbox

    async with db.execute(
        "INSERT INTO episodes (task, page_url) VALUES (?, ?) RETURNING id",
        (task, result.url_before),
    ) as cursor:
        row = await cursor.fetchone()
        episode_id = row[0]

    async with db.execute(
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
            ?,
            ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?,
            ?, ?, ?,
            ?, ?
        ) RETURNING id
        """,
        (
            episode_id,
            action.source, action.action_type.value,
            x1, y1, x2, y2,
            action.text, action.element_ref,
            action.confidence, result.agreement, result.score,
            int(result.success), result.signal,
            result.url_before, result.url_after, result.mutation_count,
            result.error, json.dumps(action.metadata),
        ),
    ) as cursor:
        row = await cursor.fetchone()
        action_log_id = row[0]

    await db.commit()
    return action_log_id
