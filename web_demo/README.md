# Web Demo

## Run

```powershell
uvicorn web_demo.app:app --reload
```

Open:

[`http://127.0.0.1:8000`](http://127.0.0.1:8000)

## What It Shows

- final answer
- retrieved top chunks
- router / planner output
- `selected_from`
- rewrite / fallback flags
- stage timings

## Recommended Environment

```powershell
$env:GENERATION_RUNTIME_PROFILE = "local_dev"
$env:OPENAI_API_KEY = "your-openai-api-key"
```
