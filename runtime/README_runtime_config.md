# Runtime Config

`runtime/runtime_config.py` 现在是 generation v3 runtime 的单一配置入口。

配置优先级：

1. CLI 参数
2. 环境变量
3. profile 默认值
4. 代码内保底默认值

## Runtime Profiles

通过 `GENERATION_RUNTIME_PROFILE` 或 `--runtime-profile` 切换：

- `local_dev`
  - `ENABLE_LLM_SURFACE_GENERATION_V3=true`
  - `ENABLE_CONTROL_CHECKS=true`
  - `ENABLE_REWRITE_V3=true`
  - `ENABLE_FALLBACK_V21=true`
  - `DEBUG_RETURN_INTERMEDIATE=true`
- `staging`
  - `ENABLE_LLM_SURFACE_GENERATION_V3=true`
  - `ENABLE_CONTROL_CHECKS=true`
  - `ENABLE_REWRITE_V3=true`
  - `ENABLE_FALLBACK_V21=true`
  - `DEBUG_RETURN_INTERMEDIATE=false`
- `production`
  - `ENABLE_LLM_SURFACE_GENERATION_V3=true`
  - `ENABLE_CONTROL_CHECKS=true`
  - `ENABLE_REWRITE_V3=true`
  - `ENABLE_FALLBACK_V21=true`
  - `DEBUG_RETURN_INTERMEDIATE=false`

默认 profile：`production`

## Environment Variables

- `GENERATION_RUNTIME_PROFILE`
  - 默认值：`production`
  - 可选值：`local_dev` / `staging` / `production`
- `ENABLE_LLM_SURFACE_GENERATION_V3`
  - 默认值：跟随 profile
  - 作用：切到 `planner_v21 -> llm_v3 -> control -> rewrite -> fallback`
- `ENABLE_CONTROL_CHECKS`
  - 默认值：跟随 profile
  - 作用：启用 mechanism name 和 how 结构校验
- `ENABLE_REWRITE_V3`
  - 默认值：跟随 profile
  - 作用：控制失败时触发 rewrite
- `ENABLE_FALLBACK_V21`
  - 默认值：跟随 profile
  - 作用：rewrite 或 llm 失败时回退到 `v21_final_answer`
- `DEBUG_RETURN_INTERMEDIATE`
  - 默认值：跟随 profile
  - 作用：在 debug 模式打印 planner / selector 中间字段
- `OPENAI_API_KEY`
  - 默认值：无
  - 作用：v3 runtime 的关键配置
- `OPENAI_BASE_URL`
  - 默认值：无
  - 作用：接 OpenAI 兼容网关时使用
- `OPENAI_MODEL`
  - 默认值：`gpt-4.1-mini`
  - 作用：surface generation / rewrite 使用的模型名
- `OPENAI_TIMEOUT_SECONDS`
  - 默认值：`60`
  - 作用：OpenAI client timeout
- `OPENAI_MAX_RETRIES`
  - 默认值：`2`
  - 作用：LLM 请求失败后的重试次数，不含首次请求

## CLI Overrides

常用覆盖参数：

- `--runtime-profile`
- `--enable-llm-surface-generation-v3` / `--disable-llm-surface-generation-v3`
- `--enable-control-checks` / `--disable-control-checks`
- `--enable-rewrite-v3` / `--disable-rewrite-v3`
- `--enable-fallback-v21` / `--disable-fallback-v21`
- `--debug-return-intermediate` / `--no-debug-return-intermediate`
- `--surface-model`
- `--surface-timeout-seconds`
- `--surface-max-retries`
- `--surface-base-url`
- `--openai-api-key`

## Local Dev Example

```powershell
$env:GENERATION_RUNTIME_PROFILE = "local_dev"
$env:OPENAI_API_KEY = "your-openai-api-key"
python chat.py --question "怎么临时推进跨部门合作？" --top-k 5 --debug
```

## Staging Example

```powershell
$env:GENERATION_RUNTIME_PROFILE = "staging"
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:OPENAI_MODEL = "gpt-4.1-mini"
python chat.py --question "怎么通过机制解决跨部门协作？" --top-k 5
```

## Production Example

```powershell
$env:GENERATION_RUNTIME_PROFILE = "production"
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:OPENAI_MODEL = "gpt-4.1-mini"
$env:OPENAI_TIMEOUT_SECONDS = "60"
$env:OPENAI_MAX_RETRIES = "2"
python chat.py --question "老板问进度时，向上汇报应该怎么压缩信息？" --top-k 5
```

## Missing Config Behavior

- `OPENAI_API_KEY` 缺失且 `ENABLE_LLM_SURFACE_GENERATION_V3=true`
  - 如果 `ENABLE_FALLBACK_V21=true`，主链路会记录错误并回到 `fallback_v21`
  - 如果 `ENABLE_FALLBACK_V21=false`，会返回清晰错误
- prompt 文件缺失
  - 会报清晰错误；若 fallback 开启，仍可回退到 `v21`
- 布尔环境变量格式错误
  - 启动阶段直接报错，例如 `maybe` 不是合法布尔值
- profile 非法
  - 启动阶段直接报错，并提示允许值

## Common Troubleshooting

- 看当前生效 profile
  - `python chat.py --question "test" --debug`
  - debug 输出里会打印 `runtime_profile`、`env_overrides`、`cli_overrides`
- 验证 API Key 是否可读
  - `python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))" `
- 强制回退旧路径
  - 设置 `ENABLE_LLM_SURFACE_GENERATION_V3=false`
- 保留 v3 但关闭 rewrite
  - 设置 `ENABLE_REWRITE_V3=false`
- 保留 rewrite 但关闭 debug 中间字段
  - 设置 `DEBUG_RETURN_INTERMEDIATE=false`
