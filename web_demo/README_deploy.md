# Shareable Demo Deploy

## Recommended Platform

Render Web Service

## Start Command

```bash
python init_db.py && uvicorn web_demo.app:app --host 0.0.0.0 --port $PORT
```

## Required Environment Variables

- `OPENAI_API_KEY`
- `GENERATION_RUNTIME_PROFILE=production`
- `ENABLE_LLM_SURFACE_GENERATION_V3=true`
- `ENABLE_CONTROL_CHECKS=true`
- `ENABLE_REWRITE_V3=true`
- `ENABLE_FALLBACK_V21=true`
- `DEBUG_RETURN_INTERMEDIATE=false`
- `DEMO_ADMIN_TOKEN`

## Optional Demo Variables

- `DEMO_TITLE`
- `DEMO_TAGLINE`
- `DEMO_DESCRIPTION`
- `DEMO_EXAMPLES`
  - format: `问题1||问题2||问题3`
- `DEMO_DEBUG_TOKEN`
- `DEMO_ADMIN_TOKEN`
- `DEMO_SHOW_DEBUG_TOGGLE=true`
- `DEMO_RATE_LIMIT_WINDOW_SECONDS=60`
- `DEMO_RATE_LIMIT_MAX_REQUESTS=12`

## Storage Notes

- `db/session.sqlite3` is used for sessions, request logs, generation logs, and feedback.
- `db/chroma` must exist and be readable in the deploy environment.
- If you want logs and feedback to persist across restarts, use a persistent disk.
- Without persistent storage, SQLite data may be lost after redeploy or instance restart.

## Render Steps

1. Push code to GitHub
2. Create a new Render Web Service
3. Point it to this repository
4. Use `deployment/render.yaml` or copy the start command above
5. Set `OPENAI_API_KEY`
6. Set `DEMO_DEBUG_TOKEN` and `DEMO_ADMIN_TOKEN`
7. Confirm `db/chroma` is available in the service image
8. Deploy and open the generated URL

## Debug Mode

If `DEMO_DEBUG_TOKEN` is set, open:

```text
https://your-demo-url?debug=1&token=your-token
```

Without a valid token, the page stays in User Mode.

## Admin View

If `DEMO_ADMIN_TOKEN` is set, open:

```text
https://your-demo-url/admin?token=your-admin-token
```

## Recommended Render Checks

- `/api/health` returns `200`
- homepage loads successfully
- `/api/config` returns examples and product copy
- `?debug=1&token=...` shows debug mode
- `/admin?token=...` loads recent requests
