# TraceAnswer

TraceAnswer is a shareable AI Q&A demo built on top of a retrieval-augmented generation pipeline with planner-based answering, controlled generation, fallback safeguards, and an optional debug view.

## Repo Layout

- `web_demo/`: FastAPI app, shareable demo UI, feedback flow, and admin view
- `runtime/`: planner, control, rewrite, selector, and runtime config modules
- `db/chroma/`: deployable Chroma index used by the demo
- `deployment/`: Render deploy config and launch checklist

## Local Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set environment variables:

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:GENERATION_RUNTIME_PROFILE = "production"
$env:DEMO_DEBUG_TOKEN = "your-debug-token"
$env:DEMO_ADMIN_TOKEN = "your-admin-token"
```

3. Initialize SQLite tables:

```bash
python init_db.py
```

4. Start the app:

```bash
uvicorn web_demo.app:app --reload
```

Open:

- User mode: `http://127.0.0.1:8000`
- Debug mode: `http://127.0.0.1:8000/?debug=1&token=your-debug-token`
- Admin view: `http://127.0.0.1:8000/admin?token=your-admin-token`

## Render Deploy

Deploy with:

- `deployment/render.yaml`
- `deployment/LAUNCH_CHECKLIST.md`
- `web_demo/README_deploy.md`

Important notes:

- `db/chroma/` should be committed if you want the demo to work without rebuilding the vector index during deploy.
- `db/session.sqlite3` should not be committed.
- If you want logs and feedback to persist across redeploys, attach a persistent disk.
