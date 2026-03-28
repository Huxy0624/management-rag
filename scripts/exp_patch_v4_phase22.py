#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("data/pipeline_candidates/v4/tagged_chunks/records.json")
DEFAULT_AUDIT_PATH = Path("data/pipeline_candidates/v4/retrieval_eval_v4/error_audit_phase22.json")
DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v4_phase22/tagged_chunks/records.json")
DEFAULT_SUMMARY_PATH = Path("data/pipeline_candidates/v4_phase22/patch_summary.json")

GLOBAL_PATCH_RULES: list[dict[str, str]] = [
    {
        "rule_bucket": "definition_negative_constraint",
        "rule": "If a chunk mainly explains process, system operation, structural causality, or action handling, do not keep answer_role=definition.",
    },
    {
        "rule_bucket": "mechanism_priority_for_structure_and_causality",
        "rule": "If the core message is structure influences behavior, rules produce incentives, or a system explains outcomes, prefer mechanism.",
    },
    {
        "rule_bucket": "solution_priority_for_actionable_handling",
        "rule": "If the core message is what to do now, how to handle, escalate, coordinate, or execute, prefer solution even when the chunk ends with a short summary.",
    },
]

PATCHES: dict[str, dict[str, Any]] = {
    "课程-21总监能力模型_2025-05-27.txt:4": {
        "answer_role": "definition",
        "intent": "none",
        "rule_bucket": "definition_not_example_action",
        "reason": "核心在定义总监这一角色，跨部门协作只是举例场景，不应被打成 solution/resource_coordination。",
    },
    "课程-25跨部门协作_2025-05-27.txt:4": {
        "answer_role": "mechanism",
        "rule_bucket": "solution_to_mechanism_for_causal_scene",
        "reason": "核心在解释跨部门协作为什么会低效与甩锅，不是在给点状操作方案。",
    },
    "课程-09刺头_2025-05-27.txt:14": {
        "answer_role": "principle",
        "rule_bucket": "solution_to_principle_for_epilogue_root_cause",
        "reason": "该段主功能已切到结语与根因回溯，强调评价失效这一上位规律，不宜继续保留 solution。",
    },
    "课程-10极品老油条_2025-05-27.txt:29": {
        "answer_role": "principle",
        "rule_bucket": "solution_to_principle_for_epilogue_root_cause",
        "reason": "该段以结语回收处理经验并上升到评价失效根因，主作用更接近原则性归纳而非步骤方案。",
    },
    "课程-11人才九宫格_2025-05-27.txt:0": {
        "answer_role": "principle",
        "rule_bucket": "solution_to_principle_for_mechanism_intro",
        "reason": "该段在引出评价失效与上升通道的治理关系，重点是机制性原则，不是即时操作建议。",
    },
    "课程-32信息通道_2025-05-27.txt:5": {
        "answer_role": "mechanism",
        "rule_bucket": "definition_to_mechanism_for_system_operation",
        "reason": "虽然出现了信息分类与定义句式，但主功能是在解释信息通道如何建设与运转，不应继续保留 definition。",
    },
    "课程-06经理_2025-05-27.txt:8": {
        "answer_role": "solution",
        "rule_bucket": "summary_to_solution_for_actionable_recap",
        "reason": "该段虽然以总结开头，但主体仍在给出处理问题、上抛跟踪与奖惩设计的动作方案，核心应回到 solution。",
    },
    "课程-27系统性思考_2025-05-27.txt:2": {
        "answer_role": "mechanism",
        "rule_bucket": "definition_to_mechanism_for_structure_behavior",
        "reason": "核心在解释结构如何影响行为、规则如何导致甩锅与协作低效，属于典型 mechanism，而不是概念定义。",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply local phase 2.2 patches on top of v4 tags.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Input v4 tagged records.")
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT_PATH, help="Phase 2.2 error audit JSON.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output patched records path.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH, help="Output patch summary path.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    records = load_json(args.input_path)
    audit = load_json(args.audit_path)

    audit_target_ids = set(audit.get("summary", {}).get("unique_target_chunk_ids", []))
    missing_from_audit = sorted(chunk_id for chunk_id in PATCHES if chunk_id not in audit_target_ids)
    if missing_from_audit:
        raise ValueError(f"Patch chunk ids missing from audit target set: {missing_from_audit}")

    patched_rows: list[dict[str, Any]] = []
    patch_log: list[dict[str, Any]] = []
    applied_counter: Counter[str] = Counter()

    for record in records:
        chunk_id = str(record.get("chunk_id", ""))
        patch = PATCHES.get(chunk_id)
        if not patch:
            patched_rows.append(record)
            continue

        updated = json.loads(json.dumps(record, ensure_ascii=False))
        tags = dict(updated.get("governance_tags_v4", {}))
        before = {
            "answer_role": tags.get("answer_role", "none"),
            "intent": tags.get("intent", "none"),
        }

        if "answer_role" in patch:
            tags["answer_role"] = patch["answer_role"]
        if "intent" in patch:
            tags["intent"] = patch["intent"]

        updated["governance_tags_v4"] = tags
        tagging_debug = dict(updated.get("tagging_debug", {}))
        tagging_debug["phase22_patch"] = {
            "rule_bucket": patch["rule_bucket"],
            "reason": patch["reason"],
            "before": before,
            "after": {
                "answer_role": tags.get("answer_role", "none"),
                "intent": tags.get("intent", "none"),
            },
        }
        updated["tagging_debug"] = tagging_debug
        patched_rows.append(updated)

        applied_counter[str(patch["rule_bucket"])] += 1
        patch_log.append(
            {
                "chunk_id": chunk_id,
                "source_file": str(updated.get("source_file", "")),
                "title": str(updated.get("title", "")),
                "before": before,
                "after": {
                    "answer_role": tags.get("answer_role", "none"),
                    "intent": tags.get("intent", "none"),
                },
                "rule_bucket": patch["rule_bucket"],
                "reason": patch["reason"],
            }
        )

    summary = {
        "base_records_path": str(args.input_path),
        "audit_path": str(args.audit_path),
        "patched_record_count": len(patch_log),
        "patched_chunk_ids": [item["chunk_id"] for item in patch_log],
        "global_patch_rules": GLOBAL_PATCH_RULES,
        "rule_bucket_distribution": dict(applied_counter),
        "patches": patch_log,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(patched_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    args.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Base records: {args.input_path}")
    print(f"Patched records: {len(patch_log)}")
    print(f"Saved patched dataset to: {args.output_path}")
    print(f"Saved patch summary to: {args.summary_path}")


if __name__ == "__main__":
    main()
