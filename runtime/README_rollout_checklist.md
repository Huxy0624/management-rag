# Rollout Checklist

## Gray Readiness

当前推荐发布顺序：

1. `local_dev`
2. `staging`
3. `production` 小流量
4. 扩大流量

## Minimum Observation Metrics

### Traffic And Path

- `v3_hit_rate`
  - 定义：`runtime_path = planner_v21_llm_v3` 的请求占比
- `legacy_hit_rate`
  - 定义：`runtime_path = legacy_ollama` 的请求占比
- `selected_from_distribution`
  - `llm_v3`
  - `rewrite_v3`
  - `fallback_v21`

### Quality And Control

- `rewrite_trigger_rate`
  - 定义：`rewrite_triggered = true` / v3 请求数
- `fallback_trigger_rate`
  - 定义：`fallback_triggered = true` / v3 请求数
- `mechanism_name_check_fail_rate`
  - 定义：`initial_control_checks.mechanism_name_check_pass = false` / mechanism_building 请求数
- `how_structure_check_fail_rate`
  - 定义：`initial_control_checks.structure_check_pass = false` / how 请求数

### Stability

- `llm_error_rate`
  - 定义：generation log 中 `success = false` 且 `provider = openai` 的占比
- `avg_latency_ms`
  - 定义：v3 generation log 平均 `latency_ms`
- `timeout_rate`
  - 定义：错误信息含 `timed out` 的占比
- `retry_rate`
  - 定义：`retry_count > 0` 的占比

### Debug And Sampling

- `selected_from_by_query_type`
  - 看 `what / why / how` 各自的 `llm_v3 / rewrite_v3 / fallback_v21` 分布
- `problematic_queries_sample`
  - 每天至少抽样：
    - 5 条 `fallback_v21`
    - 5 条 `rewrite_v3`
    - 5 条高延迟或高重试请求

## Daily / Per-Rollout Checks

### Daily

- 看 `selected_from_distribution` 是否稳定
- 看 `fallback_trigger_rate` 是否高于前一日
- 看 `llm_error_rate` 和 `avg_latency_ms` 是否恶化
- 抽样 `problematic_queries`

### Every Rollout Window

- 对比 rollout 前后：
  - `v3_hit_rate`
  - `rewrite_trigger_rate`
  - `fallback_trigger_rate`
  - `llm_error_rate`
  - `avg_latency_ms`
- 单独检查 `how` query 的结构失败率
- 单独检查 mechanism-building 的名字保留失败率

## Rollback Strategy

### Fast Rollback

目标：一键关闭 v3，回到旧路径。

```powershell
$env:ENABLE_LLM_SURFACE_GENERATION_V3 = "false"
```

或 CLI：

```powershell
python chat.py --disable-llm-surface-generation-v3
```

### Partial Rollback

只关闭 rewrite：

```powershell
$env:ENABLE_REWRITE_V3 = "false"
```

只关闭 control checks：

```powershell
$env:ENABLE_CONTROL_CHECKS = "false"
```

保留 fallback 保险丝：

```powershell
$env:ENABLE_FALLBACK_V21 = "true"
```

建议的保守局部回滚方式：

1. 保持 `ENABLE_LLM_SURFACE_GENERATION_V3=true`
2. 保持 `ENABLE_FALLBACK_V21=true`
3. 先关闭 `ENABLE_REWRITE_V3`
4. 如仍不稳，再关闭 `ENABLE_CONTROL_CHECKS`

## Suggested Rollback Thresholds

- `fallback_trigger_rate > 10%` 持续两个观察窗口
- `rewrite_trigger_rate > 25%` 持续两个观察窗口
- `llm_error_rate > 3%` 持续两个观察窗口
- `timeout_rate > 1%` 持续两个观察窗口
- `avg_latency_ms` 比基线恶化超过 `30%`
- `mechanism_name_check_fail_rate > 5%`
- `how_structure_check_fail_rate > 5%`

## Minimal Release Recommendation

第一阶段建议：

- profile：`staging`
- `ENABLE_LLM_SURFACE_GENERATION_V3=true`
- `ENABLE_CONTROL_CHECKS=true`
- `ENABLE_REWRITE_V3=true`
- `ENABLE_FALLBACK_V21=true`
- `DEBUG_RETURN_INTERMEDIATE=false`

进入 production 小流量前，至少满足：

- `fallback_trigger_rate < 5%`
- `llm_error_rate < 1%`
- `how_structure_check_fail_rate < 2%`
- `mechanism_name_check_fail_rate < 2%`
