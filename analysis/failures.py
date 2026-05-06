"""
Failure analysis report.

For each failed episode, prints all candidates ranked by score,
highlighting the selected one and flagging whether a better candidate existed.

Usage:
    python -m analysis.failures
    python -m analysis.failures --db path/to/decido.db --limit 20
"""

import argparse
import json
import sqlite3
from dataclasses import dataclass


@dataclass
class Candidate:
    rank: int
    selected: bool
    source: str
    action_type: str
    confidence: float
    agreement: float
    score: float
    text: str | None
    success: int | None  # None for non-selected


def _fetch_failed_episodes(db: sqlite3.Connection, limit: int) -> list[dict]:
    rows = db.execute(
        """
        SELECT
            e.id, e.task, e.page_url, e.created_at, e.metadata
        FROM episodes e
        JOIN candidates c ON c.episode_id = e.id AND c.selected = 1
        WHERE c.success = 0
        ORDER BY e.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def _fetch_candidates(db: sqlite3.Connection, episode_id: int) -> list[Candidate]:
    rows = db.execute(
        """
        SELECT rank, selected, source, action_type,
               confidence, agreement, score, text, success
        FROM candidates
        WHERE episode_id = ?
        ORDER BY rank
        """,
        (episode_id,),
    ).fetchall()
    return [
        Candidate(
            rank=r["rank"],
            selected=bool(r["selected"]),
            source=r["source"],
            action_type=r["action_type"],
            confidence=r["confidence"],
            agreement=r["agreement"],
            score=r["score"],
            text=r["text"],
            success=r["success"],
        )
        for r in rows
    ]


def _winner_was_best(candidates: list[Candidate]) -> bool:
    """True if the selected candidate had the highest score (no better option existed)."""
    selected = next((c for c in candidates if c.selected), None)
    if selected is None:
        return True
    return all(c.score <= selected.score for c in candidates if not c.selected)


def _format_candidate(c: Candidate) -> str:
    marker = "► SELECTED" if c.selected else "         "
    text_str = f' text="{c.text}"' if c.text else ""
    return (
        f"  {marker} [{c.rank}] {c.source:<6} {c.action_type:<6}"
        f"  conf={c.confidence:.2f}  agree={c.agreement:.3f}  score={c.score:.3f}"
        f"{text_str}"
    )


def run(db_path: str, limit: int) -> None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    episodes = _fetch_failed_episodes(con, limit)

    if not episodes:
        print("No failed episodes found.")
        return

    print(f"{'='*70}")
    print(f"  FAILURE REPORT  —  {len(episodes)} failed episode(s)")
    print(f"{'='*70}\n")

    for ep in episodes:
        metadata = json.loads(ep["metadata"] or "{}")
        candidates = _fetch_candidates(con, ep["id"])
        missed_better = not _winner_was_best(candidates)

        print(f"Episode {ep['id']}  {ep['created_at']}")
        print(f"  Task:    {ep['task']}")
        print(f"  URL:     {ep['page_url']}")

        dom_ms = metadata.get("dom_latency_ms", "?")
        vis_ms = metadata.get("vision_latency_ms", "?")
        dom_fail = metadata.get("dom_failed", False)
        vis_fail = metadata.get("vision_failed", False)
        print(
            f"  Agents:  dom={dom_ms}ms {'(FAILED)' if dom_fail else ''}"
            f"  vision={vis_ms}ms {'(FAILED)' if vis_fail else ''}"
        )

        if missed_better:
            print("  ⚠  A higher-scoring candidate was NOT selected (scorer error?)")

        print(f"  Candidates ({len(candidates)} total):")
        for c in candidates:
            print(_format_candidate(c))
        print()

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decido failure analysis")
    parser.add_argument("--db", default="decido.db", help="Path to SQLite database")
    parser.add_argument("--limit", type=int, default=10, help="Max episodes to show")
    args = parser.parse_args()
    run(args.db, args.limit)
