CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task        TEXT    NOT NULL,
    page_url    TEXT    NOT NULL,
    metadata    TEXT    NOT NULL DEFAULT '{}',
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    rank            INTEGER NOT NULL,   -- position in scored list (1 = best)
    selected        INTEGER NOT NULL,   -- 1 if this candidate was executed, else 0

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

    -- execution outcome: only populated for the selected candidate
    success         INTEGER,
    signal          TEXT,
    url_before      TEXT    NOT NULL,
    url_after       TEXT,
    mutation_count  INTEGER,
    error           TEXT,

    metadata        TEXT    NOT NULL DEFAULT '{}',
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_candidates_episode   ON candidates(episode_id);
CREATE INDEX IF NOT EXISTS idx_candidates_selected  ON candidates(selected);
CREATE INDEX IF NOT EXISTS idx_candidates_source    ON candidates(source);
