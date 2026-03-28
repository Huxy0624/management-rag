-- Common SQLite queries for inspecting db/session.sqlite3
-- Replace :session_id with a real session ID in your database tool.

-- 1. View all sessions ordered by latest activity
SELECT
    session_id,
    title,
    user_id,
    status,
    created_at,
    updated_at
FROM sessions
ORDER BY updated_at DESC;


-- 2. View all messages for one session ordered by turn
SELECT
    message_id,
    session_id,
    role,
    turn_index,
    content,
    created_at
FROM messages
WHERE session_id = :session_id
ORDER BY turn_index ASC, created_at ASC;


-- 3. View retrieval logs for one session
SELECT
    retrieval_id,
    session_id,
    message_id,
    raw_query,
    retrieval_query,
    top_k,
    context_length,
    rerank_applied,
    retrieved_items_json,
    created_at
FROM retrieval_logs
WHERE session_id = :session_id
ORDER BY created_at ASC;


-- 4. View generation logs for one session
SELECT
    generation_id,
    session_id,
    message_id,
    provider,
    model_name,
    prompt_length,
    latency_ms,
    retry_count,
    success,
    error_message,
    created_at,
    prompt_text,
    answer
FROM generation_logs
WHERE session_id = :session_id
ORDER BY created_at ASC;


-- 5. View the latest 20 failed generations
SELECT
    generation_id,
    session_id,
    message_id,
    provider,
    model_name,
    latency_ms,
    retry_count,
    success,
    error_message,
    created_at
FROM generation_logs
WHERE success = 0
ORDER BY created_at DESC
LIMIT 20;


-- 6. Average generation latency for successful requests
SELECT
    ROUND(AVG(latency_ms), 2) AS avg_latency_ms
FROM generation_logs
WHERE success = 1
  AND latency_ms IS NOT NULL;


-- 7. Success / failure count summary
SELECT
    CASE success
        WHEN 1 THEN 'success'
        ELSE 'failure'
    END AS result,
    COUNT(*) AS count
FROM generation_logs
GROUP BY success
ORDER BY result;


-- 8. One-session full joined timeline
SELECT
    s.session_id,
    s.title,
    m.message_id,
    m.role,
    m.turn_index,
    m.content,
    rl.retrieval_id,
    rl.top_k,
    rl.context_length,
    rl.rerank_applied,
    gl.generation_id,
    gl.provider,
    gl.model_name,
    gl.prompt_length,
    gl.latency_ms,
    gl.retry_count,
    gl.success,
    gl.error_message,
    m.created_at
FROM sessions s
LEFT JOIN messages m
    ON s.session_id = m.session_id
LEFT JOIN retrieval_logs rl
    ON m.message_id = rl.message_id
LEFT JOIN generation_logs gl
    ON m.message_id = gl.message_id
WHERE s.session_id = :session_id
ORDER BY m.turn_index ASC, m.created_at ASC;


-- 9. Recent sessions with message / retrieval / generation counts
SELECT
    s.session_id,
    s.title,
    s.status,
    s.updated_at,
    COUNT(DISTINCT m.message_id) AS message_count,
    COUNT(DISTINCT rl.retrieval_id) AS retrieval_count,
    COUNT(DISTINCT gl.generation_id) AS generation_count
FROM sessions s
LEFT JOIN messages m
    ON s.session_id = m.session_id
LEFT JOIN retrieval_logs rl
    ON s.session_id = rl.session_id
LEFT JOIN generation_logs gl
    ON s.session_id = gl.session_id
GROUP BY s.session_id, s.title, s.status, s.updated_at
ORDER BY s.updated_at DESC;


-- 10. Check whether one session has complete logs
SELECT
    s.session_id,
    s.title,
    COUNT(DISTINCT CASE WHEN m.role = 'user' THEN m.message_id END) AS user_messages,
    COUNT(DISTINCT CASE WHEN m.role = 'assistant' THEN m.message_id END) AS assistant_messages,
    COUNT(DISTINCT rl.retrieval_id) AS retrieval_logs,
    COUNT(DISTINCT gl.generation_id) AS generation_logs
FROM sessions s
LEFT JOIN messages m
    ON s.session_id = m.session_id
LEFT JOIN retrieval_logs rl
    ON s.session_id = rl.session_id
LEFT JOIN generation_logs gl
    ON s.session_id = gl.session_id
WHERE s.session_id = :session_id
GROUP BY s.session_id, s.title;
