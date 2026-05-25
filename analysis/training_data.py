"""
Extract training data from the candidates table for the v3 scoring model.

Each row is one candidate from an executed episode. The label is whether
that candidate would have succeeded if selected (only known for the
actually-selected candidate — the rest are unlabeled).

For the selected candidate: success is directly observed.
For non-selected candidates: success is NULL (unknown). These can be used
for contrastive learning (selected-vs-not) but not for direct supervision.

Usage:
    python -m analysis.training_data --db decido.db --out training.csv
    python -m analysis.training_data --db decido.db --selected-only --out training.csv
"""

import argparse
import csv
import json
import sqlite3
from pathlib import Path


FEATURES = [
    "episode_id",
    "candidate_id",
    "selected",
    "source",
    "action_type",
    "agreement",
    "score",
    "keyword_match",
    "type_coherence",
    "bbox_area",
    "bbox_center_x",
    "bbox_center_y",
    "has_text",
    "text_length",
    "has_element_ref",
    "mutation_count",
    "session_step_index",
    "session_prior_failures",
    "session_same_action_type_failures",
    "success",
    "signal",
]


def extract(db_path: str, selected_only: bool) -> list[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    where = "AND c.selected = 1" if selected_only else ""
    rows = con.execute(
        f"""
        SELECT
            c.id             AS candidate_id,
            c.episode_id,
            c.selected,
            c.source,
            c.action_type,
            c.agreement,
            c.score,
            c.metadata       AS candidate_meta,
            c.bbox_x1, c.bbox_y1, c.bbox_x2, c.bbox_y2,
            c.text,
            c.element_ref,
            c.success,
            c.signal,
            c.mutation_count,
            e.metadata       AS episode_meta
        FROM candidates c
        JOIN episodes e ON e.id = c.episode_id
        WHERE 1=1 {where}
        ORDER BY c.episode_id, c.rank
        """,
    ).fetchall()

    # Build session context: for each episode, find the step and prior history
    step_context = {}
    try:
        step_rows = con.execute(
            """
            SELECT
                s.episode_id,
                s.step_index,
                s.session_id
            FROM steps s
            ORDER BY s.session_id, s.step_index
            """,
        ).fetchall()

        # Group steps by session for prior-failure counting
        sessions: dict[int, list] = {}
        ep_to_step: dict[int, dict] = {}
        for sr in step_rows:
            sid = sr["session_id"]
            if sid not in sessions:
                sessions[sid] = []
            sessions[sid].append(sr)
            ep_to_step[sr["episode_id"]] = {
                "step_index": sr["step_index"],
                "session_id": sid,
            }

        # For each episode, count prior failures in its session
        for ep_id, info in ep_to_step.items():
            sid = info["session_id"]
            step_idx = info["step_index"]
            prior_steps = sessions.get(sid, [])

            # Get action outcomes for prior steps via candidates
            prior_episode_ids = [
                s["episode_id"] for s in prior_steps if s["step_index"] < step_idx
            ]
            prior_failures = 0
            same_type_failures: dict[str, int] = {}

            if prior_episode_ids:
                placeholders = ",".join("?" * len(prior_episode_ids))
                prior_cands = con.execute(
                    f"""
                    SELECT action_type, success
                    FROM candidates
                    WHERE episode_id IN ({placeholders}) AND selected = 1
                    """,
                    prior_episode_ids,
                ).fetchall()
                for pc in prior_cands:
                    if pc["success"] == 0:
                        prior_failures += 1
                        at = pc["action_type"]
                        same_type_failures[at] = same_type_failures.get(at, 0) + 1

            step_context[ep_id] = {
                "step_index": step_idx,
                "prior_failures": prior_failures,
                "same_type_failures": same_type_failures,
            }
    except sqlite3.OperationalError:
        pass

    records = []
    for r in rows:
        cand_meta = json.loads(r["candidate_meta"] or "{}")
        ep_ctx = step_context.get(r["episode_id"], {})

        bbox_w = r["bbox_x2"] - r["bbox_x1"]
        bbox_h = r["bbox_y2"] - r["bbox_y1"]

        record = {
            "episode_id": r["episode_id"],
            "candidate_id": r["candidate_id"],
            "selected": r["selected"],
            "source": r["source"],
            "action_type": r["action_type"],
            "agreement": r["agreement"],
            "score": r["score"],
            "keyword_match": cand_meta.get("keyword_match", 0.0),
            "type_coherence": cand_meta.get("type_coherence", 0.0),
            "bbox_area": bbox_w * bbox_h,
            "bbox_center_x": (r["bbox_x1"] + r["bbox_x2"]) / 2,
            "bbox_center_y": (r["bbox_y1"] + r["bbox_y2"]) / 2,
            "has_text": 1 if r["text"] else 0,
            "text_length": len(r["text"]) if r["text"] else 0,
            "has_element_ref": 1 if r["element_ref"] else 0,
            "mutation_count": r["mutation_count"] or 0,
            "session_step_index": ep_ctx.get("step_index", -1),
            "session_prior_failures": ep_ctx.get("prior_failures", 0),
            "session_same_action_type_failures": ep_ctx.get(
                "same_type_failures", {}
            ).get(r["action_type"], 0),
            "success": r["success"],
            "signal": r["signal"],
        }
        records.append(record)

    con.close()
    return records


def run(db_path: str, out_path: str, selected_only: bool) -> None:
    records = extract(db_path, selected_only)

    if not records:
        print("No candidate data found.")
        return

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURES)
        writer.writeheader()
        writer.writerows(records)

    labeled = sum(1 for r in records if r["success"] is not None)
    positive = sum(1 for r in records if r["success"] == 1)
    print(f"Extracted {len(records)} candidates ({labeled} labeled, {positive} positive)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract training data for v3 scorer")
    parser.add_argument("--db", default="decido.db")
    parser.add_argument("--out", default="training.csv")
    parser.add_argument("--selected-only", action="store_true",
                        help="Only export selected (executed) candidates — all labeled")
    args = parser.parse_args()
    run(args.db, args.out, args.selected_only)
