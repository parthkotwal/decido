"""
Mutation count analysis.

Shows the distribution of mutation_count values across action types and
success signals — used to tune MUTATION_THRESHOLD in executor.py.

Usage:
    python -m analysis.mutations
    python -m analysis.mutations --db path/to/decido.db
"""

import argparse
import sqlite3


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    total = con.execute(
        "SELECT COUNT(*) FROM candidates WHERE selected = 1"
    ).fetchone()[0]

    if total == 0:
        print("No executed candidates in DB yet — run some tasks first.")
        return

    print(f"\nAnalysing {total} executed action(s) in {db_path}\n")

    # ── 1. Global mutation count distribution ─────────────────────────────────
    _section("Global mutation_count distribution (selected candidates)")
    rows = con.execute(
        """
        SELECT
            mutation_count,
            COUNT(*)            AS n,
            SUM(success)        AS successes,
            ROUND(AVG(success) * 100, 1) AS success_rate
        FROM candidates
        WHERE selected = 1
        GROUP BY mutation_count
        ORDER BY mutation_count
        """
    ).fetchall()

    print(f"  {'mutations':>10}  {'n':>5}  {'successes':>9}  {'success%':>9}")
    for r in rows:
        print(
            f"  {r['mutation_count']:>10}  {r['n']:>5}  "
            f"{r['successes']:>9}  {r['success_rate']:>8}%"
        )

    # ── 2. By action type ─────────────────────────────────────────────────────
    _section("Average mutation_count by action_type")
    rows = con.execute(
        """
        SELECT
            action_type,
            COUNT(*)                        AS n,
            ROUND(AVG(mutation_count), 1)   AS avg_mutations,
            MIN(mutation_count)             AS min_mutations,
            MAX(mutation_count)             AS max_mutations,
            ROUND(AVG(success) * 100, 1)    AS success_rate
        FROM candidates
        WHERE selected = 1
        GROUP BY action_type
        ORDER BY avg_mutations DESC
        """
    ).fetchall()

    print(
        f"  {'action_type':>12}  {'n':>5}  {'avg':>6}  "
        f"{'min':>5}  {'max':>5}  {'success%':>9}"
    )
    for r in rows:
        print(
            f"  {r['action_type']:>12}  {r['n']:>5}  {r['avg_mutations']:>6}  "
            f"{r['min_mutations']:>5}  {r['max_mutations']:>5}  {r['success_rate']:>8}%"
        )

    # ── 3. By success signal ───────────────────────────────────────────────────
    _section("mutation_count by success signal")
    rows = con.execute(
        """
        SELECT
            COALESCE(signal, 'none')        AS signal,
            COUNT(*)                        AS n,
            ROUND(AVG(mutation_count), 1)   AS avg_mutations,
            MIN(mutation_count)             AS min_mutations,
            MAX(mutation_count)             AS max_mutations
        FROM candidates
        WHERE selected = 1
        GROUP BY signal
        ORDER BY avg_mutations DESC
        """
    ).fetchall()

    print(
        f"  {'signal':>16}  {'n':>5}  {'avg':>6}  {'min':>5}  {'max':>5}"
    )
    for r in rows:
        print(
            f"  {r['signal']:>16}  {r['n']:>5}  {r['avg_mutations']:>6}  "
            f"{r['min_mutations']:>5}  {r['max_mutations']:>5}"
        )

    # ── 4. Failures with low mutation count (likely threshold victims) ─────────
    _section("Failed actions with mutation_count 1–5 (likely threshold victims)")
    rows = con.execute(
        """
        SELECT
            e.task,
            c.action_type,
            c.source,
            c.mutation_count,
            c.score
        FROM candidates c
        JOIN episodes e ON e.id = c.episode_id
        WHERE c.selected = 1
          AND c.success = 0
          AND c.mutation_count BETWEEN 1 AND 5
          AND c.signal IS NULL
        ORDER BY c.mutation_count DESC
        """
    ).fetchall()

    if not rows:
        print("  None found — threshold looks okay for current data.")
    else:
        print(f"  {'mutations':>10}  {'action':>8}  {'source':>7}  {'score':>6}  task")
        for r in rows:
            task_preview = r["task"][:45] + "…" if len(r["task"]) > 45 else r["task"]
            print(
                f"  {r['mutation_count']:>10}  {r['action_type']:>8}  "
                f"{r['source']:>7}  {r['score']:>6.3f}  {task_preview}"
            )

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decido mutation count analysis")
    parser.add_argument("--db", default="decido.db", help="Path to SQLite database")
    args = parser.parse_args()
    run(args.db)
