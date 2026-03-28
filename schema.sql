PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    user_id TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted')),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS retrieval_logs (
    retrieval_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    raw_query TEXT NOT NULL,
    retrieval_query TEXT NOT NULL,
    top_k INTEGER NOT NULL,
    retrieved_items_json TEXT NOT NULL,
    context_length INTEGER NOT NULL DEFAULT 0,
    rerank_applied INTEGER NOT NULL DEFAULT 0 CHECK (rerank_applied IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS generation_logs (
    generation_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    prompt_length INTEGER NOT NULL,
    answer TEXT,
    latency_ms INTEGER,
    retry_count INTEGER NOT NULL DEFAULT 0,
    success INTEGER NOT NULL DEFAULT 1 CHECK (success IN (0, 1)),
    error_message TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (message_id) REFERENCES messages(message_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session_created
ON messages (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_messages_session_turn
ON messages (session_id, turn_index);

CREATE INDEX IF NOT EXISTS idx_retrieval_logs_session_created
ON retrieval_logs (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_retrieval_logs_message
ON retrieval_logs (message_id);

CREATE INDEX IF NOT EXISTS idx_generation_logs_session_created
ON generation_logs (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_generation_logs_message
ON generation_logs (message_id);

CREATE TABLE IF NOT EXISTS demo_request_logs (
    request_id TEXT PRIMARY KEY,
    session_id TEXT,
    question TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 1 CHECK (success IN (0, 1)),
    selected_from TEXT,
    fallback_triggered INTEGER NOT NULL DEFAULT 0 CHECK (fallback_triggered IN (0, 1)),
    total_latency_ms INTEGER,
    error_message TEXT,
    client_ip TEXT,
    user_mode TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS demo_feedback (
    feedback_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    session_id TEXT,
    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
    comment TEXT,
    client_ip TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_demo_request_logs_created
ON demo_request_logs (created_at);

CREATE INDEX IF NOT EXISTS idx_demo_request_logs_session
ON demo_request_logs (session_id);

CREATE INDEX IF NOT EXISTS idx_demo_feedback_request
ON demo_feedback (request_id);
