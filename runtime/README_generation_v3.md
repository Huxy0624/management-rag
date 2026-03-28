# Generation V3 Runtime

## Call Order

`chat.py` 在启用 `--enable-llm-surface-generation-v3` 后会按下面顺序执行：

1. `retrieve_context()` 取回并重排检索结果
2. `runtime.query_router.route_query()` 判断 `query_type` 和 `router_decision`
3. `runtime.planner_runtime.build_planner_context()` 构造 `planner_output_v21`
4. `runtime.llm_surface_runtime.generate_surface_answer()` 进行 v3 surface generation
5. `runtime.control_layer.build_control_checks()` 检查 mechanism name 与 how 结构
6. `runtime.rewrite_runtime.rewrite_answer()` 在控制失败时触发 rewrite
7. `runtime.answer_selector.select_answer()` 选择 `llm_v3 / rewrite_v3 / fallback_v21`

## Feature Flags

- `--enable-llm-surface-generation-v3`
- `--disable-llm-surface-generation-v3`
- `--enable-control-checks`
- `--disable-control-checks`
- `--enable-rewrite-v3`
- `--disable-rewrite-v3`
- `--enable-fallback-v21`
- `--disable-fallback-v21`
- `--debug-return-intermediate`

## Debug Fields

当同时打开 `--debug --debug-return-intermediate` 时，CLI 会打印：

- `query_type`
- `router_decision`
- `planner_output_v21`
- `llm_answer`
- `final_selected_answer`
- `selected_from`
- `mechanism_name_check_pass`
- `action_steps_match_count`
- `rewrite_triggered`
- `fallback_triggered`

## Logging

`generation_logs.metadata_json` 现在会记录：

- `selected_from`
- `query_type`
- `rewrite_triggered`
- `fallback_triggered`
- `initial_control_checks`
- `final_control_checks`

## Rollout Suggestion

1. Internal/debug：只在 `--enable-llm-surface-generation-v3 --debug-return-intermediate` 下验证。
2. Small traffic：保持 `enable_rewrite_v3=true`、`enable_fallback_v21=true`，观察 `selected_from` 分布。
3. Stabilize：按 `query_type` 统计 rewrite/fallback 触发率。
4. Expand：当 fallback 触发率稳定且集中在少数长尾 query 后，再扩大流量。
