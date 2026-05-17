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

-- ── v3: multi-step sessions ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    task                TEXT    NOT NULL,
    start_url           TEXT    NOT NULL,

    -- one of: running | complete | failed
    status              TEXT    NOT NULL DEFAULT 'running',

    -- what ended the session (NULL while running)
    -- success | action_failure | cascade | loop | premature_termination |
    -- memory_miss | step_limit | task_ambiguity
    termination_reason  TEXT,

    step_count          INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
    ended_at            TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    episode_id      INTEGER NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    step_index      INTEGER NOT NULL,   -- 0-based position within the session
    page_url        TEXT    NOT NULL,   -- URL at the start of this step
    axtree_hash     TEXT    NOT NULL,   -- hash of page state, used for loop detection
    checkpoint      INTEGER NOT NULL DEFAULT 0,  -- 1 if this step was saved for rollback

    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_steps_session    ON steps(session_id);
CREATE INDEX IF NOT EXISTS idx_steps_episode    ON steps(episode_id);

CREATE TABLE IF NOT EXISTS rollbacks (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id          INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

    triggered_at_step   INTEGER NOT NULL,   -- step index that caused the rollback
    rolled_back_to_step INTEGER NOT NULL,   -- checkpoint step we restored to

    -- why rollback was triggered
    -- "cascade" (automatic) | "manual" (API request)
    reason              TEXT    NOT NULL,

    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_rollbacks_session ON rollbacks(session_id);
