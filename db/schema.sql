CREATE TABLE IF NOT EXISTS episodes (
    id          BIGSERIAL PRIMARY KEY,
    task        TEXT        NOT NULL,
    page_url    TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS action_logs (
    id              BIGSERIAL PRIMARY KEY,
    episode_id      BIGINT      NOT NULL REFERENCES episodes(id) ON DELETE CASCADE,

    -- which agent proposed this action
    source          TEXT        NOT NULL,   -- "dom" | "vision"
    action_type     TEXT        NOT NULL,
    bbox_x1         REAL        NOT NULL,
    bbox_y1         REAL        NOT NULL,
    bbox_x2         REAL        NOT NULL,
    bbox_y2         REAL        NOT NULL,
    text            TEXT,
    element_ref     TEXT,

    -- scorer features (training signal for v3)
    confidence      REAL        NOT NULL,
    agreement       REAL        NOT NULL,
    score           REAL        NOT NULL,

    -- execution outcome
    success         BOOLEAN     NOT NULL,
    signal          TEXT,                   -- "url_change" | "dom_mutation" | NULL
    url_before      TEXT        NOT NULL,
    url_after       TEXT        NOT NULL,
    mutation_count  INT         NOT NULL DEFAULT 0,
    error           TEXT,

    metadata        JSONB       NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- useful for querying training data by agent and outcome
CREATE INDEX IF NOT EXISTS idx_action_logs_source  ON action_logs(source);
CREATE INDEX IF NOT EXISTS idx_action_logs_success ON action_logs(success);
CREATE INDEX IF NOT EXISTS idx_action_logs_episode ON action_logs(episode_id);
