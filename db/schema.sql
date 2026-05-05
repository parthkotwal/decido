CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task        TEXT    NOT NULL,
    page_url    TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS action_logs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    source          TEXT    NOT NULL,
    action_type     TEXT    NOT NULL,
    bbox_x1         REAL    NOT NULL,
    bbox_y1         REAL    NOT NULL,
    bbox_x2         REAL    NOT NULL,
    bbox_y2         REAL    NOT NULL,
    text            TEXT,
    element_ref     TEXT,

    confidence      REAL    NOT NULL,
    agreement       REAL    NOT NULL,
    score           REAL    NOT NULL,

    success         INTEGER NOT NULL,   -- 0 | 1 (SQLite has no boolean type)
    signal          TEXT,
    url_before      TEXT    NOT NULL,
    url_after       TEXT    NOT NULL,
    mutation_count  INTEGER NOT NULL DEFAULT 0,
    error           TEXT,

    metadata        TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_action_logs_source   ON action_logs(source);
CREATE INDEX IF NOT EXISTS idx_action_logs_success  ON action_logs(success);
CREATE INDEX IF NOT EXISTS idx_action_logs_episode  ON action_logs(episode_id);
