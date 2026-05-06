import json
import aiosqlite

from core.scorer import ScoredAction
from execution.executor import ExecutionResult

DB_PATH = "decido.db"


async def init_db(db: aiosqlite.Connection) -> None:
    with open("db/schema.sql") as f:
        await db.executescript(f.read())
    await db.commit()


async def log_candidates(
    db: aiosqlite.Connection,
    task: str,
    scored: list[ScoredAction],
    best: ScoredAction,
    result: ExecutionResult,
    episode_metadata: dict | None = None,
) -> int:
    """
    Persist all scored candidates for an episode.

    The selected candidate (best) gets execution outcome columns populated.
    All others get NULL for those columns.

    Returns the episode id.
    """
    async with db.execute(
        "INSERT INTO episodes (task, page_url, metadata) VALUES (?, ?, ?) RETURNING id",
        (task, result.url_before, json.dumps(episode_metadata or {})),
    ) as cursor:
        row = await cursor.fetchone()
        episode_id = row[0]

    for rank, sc in enumerate(scored, start=1):
        action = sc.action
        x1, y1, x2, y2 = action.bbox
        is_selected = sc is best

        await db.execute(
            """
            INSERT INTO candidates (
                episode_id,
                rank, selected,
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
                ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?
            )
            """,
            (
                episode_id,
                rank, int(is_selected),
                action.source, action.action_type.value,
                x1, y1, x2, y2,
                action.text, action.element_ref,
                sc.confidence, sc.agreement, sc.score,
                int(result.success) if is_selected else None,
                result.signal if is_selected else None,
                result.url_before,
                result.url_after if is_selected else None,
                result.mutation_count if is_selected else None,
                result.error if is_selected else None,
                json.dumps(action.metadata),
            ),
        )

    await db.commit()
    return episode_id
