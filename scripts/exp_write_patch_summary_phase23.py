#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import exp_validate_retrieval_v4_phase23 as phase23


OUTPUT_PATH = Path("data/pipeline_candidates/v4_phase23/patch_summary_phase23.json")


def main() -> None:
    summary = {
        "base_records_path": "data/pipeline_candidates/v4_phase22/tagged_chunks/records.json",
        "base_report_path": "data/pipeline_candidates/v4_phase22/retrieval_eval_v4/report.json",
        "phase23_report_path": "data/pipeline_candidates/v4_phase23/report.json",
        "change_scope": "rerank_only",
        "strong_primary_criteria": phase23.STRONG_PRIMARY_CRITERIA,
        "penalty_decay_policy": phase23.PENALTY_DECAY_POLICY,
        "mismatch_tier_policy": phase23.MISMATCH_TIER_POLICY,
        "meta_like_heuristic": {
            "description": "Derived heuristic only, not persisted into formal schema.",
            "keywords": list(phase23.META_LIKE_KEYWORDS),
            "effect": [
                "weaken definition negative constraint on meta-like chunks",
                "reduce principle/mechanism/definition mismatch penalties",
                "keep penalties but avoid one-shot heavy suppression",
            ],
        },
        "applicable_range": {
            "rerank_pool_only": True,
            "top_m": 15,
            "schema_changes": False,
            "record_relabeling": False,
            "mainline_changes": False,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved phase23 patch summary to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
