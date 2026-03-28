## Database View Guide

This project stores chat state and runtime logs in `db/session.sqlite3`.
You can open that file directly with any SQLite viewer such as DB Browser for SQLite, DBeaver, DataGrip, TablePlus, or the SQLite extension in VS Code.

Use `queries.sql` as a ready-to-run query collection inside your database client.

## Tables

### `sessions`
Purpose:
- One row per conversation session.
- This is the top-level container for all messages and logs.

Important fields:
- `session_id`: the primary identifier for one conversation.
- `title`: usually the first question or a short session label.
- `user_id`: optional, reserved for multi-user support later.
- `status`: current session state such as `active`.
- `created_at` / `updated_at`: session timeline.

How to inspect:
- Start with this table to find the session you want.
- Sort by `updated_at` descending to find the most recent conversations.

### `messages`
Purpose:
- Stores the actual chat messages.
- Includes user, assistant, and system messages.

Important fields:
- `message_id`: unique ID for one message.
- `session_id`: links the message back to a session.
- `role`: `user`, `assistant`, or `system`.
- `content`: the message body.
- `turn_index`: the logical order of the conversation.
- `created_at`: message timestamp.

How to inspect:
- Filter by `session_id`.
- Order by `turn_index` to reconstruct the conversation.

### `retrieval_logs`
Purpose:
- Stores the RAG retrieval record for one user question.
- Useful for debugging search quality and rerank behavior.

Important fields:
- `retrieval_id`: unique retrieval record ID.
- `session_id`: links retrieval to a session.
- `message_id`: usually points to the related user message.
- `raw_query`: original user question.
- `retrieval_query`: current retrieval query text.
- `top_k`: how many chunks were requested after rerank.
- `retrieved_items_json`: JSON text containing retrieved chunk metadata.
- `context_length`: total context size recorded for retrieval.
- `rerank_applied`: whether rerank was used.
- `created_at`: retrieval timestamp.

How to inspect:
- Filter by `session_id`.
- Check whether `retrieved_items_json` contains the expected `source`, `chunk_id`, `vector_score`, and `rerank_score`.

### `generation_logs`
Purpose:
- Stores the model generation record for one answer.
- Useful for prompt debugging, latency monitoring, retry analysis, and failure tracking.

Important fields:
- `generation_id`: unique generation record ID.
- `session_id`: links generation to a session.
- `message_id`: usually points to the related user message.
- `provider`: current provider, for example `ollama`.
- `model_name`: current model, for example `qwen2.5:7b`.
- `prompt_text`: the full prompt sent to the model.
- `prompt_length`: prompt size in characters.
- `answer`: generated answer text.
- `latency_ms`: generation latency.
- `retry_count`: how many retries happened before success or failure.
- `success`: `1` means success, `0` means failure.
- `error_message`: failure reason when `success = 0`.
- `created_at`: generation timestamp.

How to inspect:
- Filter by `session_id`.
- Check prompt length, latency, retries, and whether `success = 1`.

## How To Trace One Full QA Turn

Recommended order:

1. Open `sessions` and find the target `session_id`.
2. Open `messages` filtered by that `session_id`.
3. Find the user message for the turn you care about.
4. Open `retrieval_logs` filtered by the same `session_id`.
5. Open `generation_logs` filtered by the same `session_id`.

Practical join logic:
- `sessions.session_id = messages.session_id`
- `messages.message_id = retrieval_logs.message_id`
- `messages.message_id = generation_logs.message_id`

This lets you move from:
- one session
- to one user message
- to the retrieval data
- to the final generation record

## How To Judge Whether One QA Turn Was Fully Logged

For a normal RAG turn, a complete write usually means:
- a `sessions` row exists
- one `user` message exists
- one `assistant` message exists
- one `retrieval_logs` row exists
- one `generation_logs` row exists

Typical successful pattern:
- `messages` has both `user` and `assistant`
- `retrieval_logs.rerank_applied = 1`
- `generation_logs.success = 1`
- `generation_logs.answer` is not empty

For `--no-rag` mode:
- `retrieval_logs` may be absent
- `generation_logs` should still exist
- `messages` should still include both `user` and `assistant`

## What To Focus On During Analysis

If you are checking conversation integrity:
- `sessions.updated_at`
- `messages.turn_index`
- `messages.role`

If you are checking retrieval quality:
- `retrieval_logs.raw_query`
- `retrieval_logs.top_k`
- `retrieved_items_json`
- `context_length`
- `rerank_applied`

If you are checking generation quality:
- `generation_logs.prompt_length`
- `generation_logs.latency_ms`
- `generation_logs.retry_count`
- `generation_logs.success`
- `generation_logs.error_message`
- `generation_logs.prompt_text`
- `generation_logs.answer`

## Suggested Workflow In A SQLite Client

1. Open `db/session.sqlite3`.
2. Run the first query in `queries.sql` to list recent sessions.
3. Copy one `session_id`.
4. Replace `:session_id` in the session-specific queries with that value.
5. Inspect messages, retrieval logs, and generation logs together.

If one answer looks wrong:
- check whether retrieval returned the wrong chunks
- check whether prompt length is too large
- check whether generation failed or retried
- compare `retrieved_items_json` with `answer`
