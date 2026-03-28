#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import exp_chunk_cleaned_v2 as chunker
import exp_clean_articles_v2 as cleaner
import exp_extract_insights_v2 as insighter
import exp_tag_chunks_v1_1 as tagger


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_ROOT = Path("data/pipeline_candidates/v1")
DEFAULT_MIN_CHARS = 200
DEFAULT_TARGET_CHARS = 360
DEFAULT_MAX_CHARS = 500
DEFAULT_MIN_SENTENCES = 3
DEFAULT_MAX_SENTENCES = 8
DEFAULT_OVERLAP_RATIO = 0.15
DEFAULT_MAX_INSIGHTS_PER_CHUNK = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full offline pipeline candidates generation for all raw articles.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Raw article directory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root output directory for candidate dataset.")
    parser.add_argument("--stage", choices=["all", "clean", "chunk", "insight", "tag", "summary"], default="all", help="Run one stage or the full pipeline.")
    parser.add_argument("--limit", type=int, help="Optional limit for debug runs.")
    parser.add_argument("--source-file", type=str, help="Optional source_file substring for targeted reruns.")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--min-sentences", type=int, default=DEFAULT_MIN_SENTENCES)
    parser.add_argument("--max-sentences", type=int, default=DEFAULT_MAX_SENTENCES)
    parser.add_argument("--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO)
    parser.add_argument("--max-insights-per-chunk", type=int, default=DEFAULT_MAX_INSIGHTS_PER_CHUNK)
    return parser.parse_args()


def count_supported_raw_files(input_dir: Path) -> list[Path]:
    return list(cleaner.iter_source_files(input_dir))


def stage_paths(output_root: Path) -> dict[str, Path]:
    return {
        "clean_records": output_root / "cleaned" / "records.json",
        "clean_stats": output_root / "cleaned" / "stats.json",
        "chunk_records": output_root / "chunks" / "records.json",
        "chunk_stats": output_root / "chunks" / "stats.json",
        "insight_records": output_root / "insights" / "records.json",
        "insight_stats": output_root / "insights" / "stats.json",
        "tag_records": output_root / "tagged_chunks" / "records.json",
        "tag_stats": output_root / "tagged_chunks" / "stats.json",
        "summary": output_root / "summary.json",
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def unique_source_count(records: list[dict[str, Any]]) -> int:
    return len({str(record.get("source_file", "")) for record in records if str(record.get("source_file", ""))})


def preview_source_names(files: list[Path], max_items: int = 10) -> list[str]:
    names = [path.name for path in files[:max_items]]
    if len(files) > max_items:
        names.append(f"... (+{len(files) - max_items} more)")
    return names


def apply_filters(records: list[dict[str, Any]], source_file: str | None, limit: int | None) -> list[dict[str, Any]]:
    filtered = records
    if source_file:
        filtered = [record for record in filtered if source_file in str(record.get("source_file", ""))]
    if limit is None:
        return filtered

    kept_sources: list[str] = []
    for record in filtered:
        current = str(record.get("source_file", ""))
        if current and current not in kept_sources:
            kept_sources.append(current)
        if len(kept_sources) >= limit:
            break

    kept_set = set(kept_sources)
    return [record for record in filtered if str(record.get("source_file", "")) in kept_set]


def merge_by_source_file(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not incoming:
        return existing

    replace_sources = {str(item.get("source_file", "")) for item in incoming}
    preserved = [item for item in existing if str(item.get("source_file", "")) not in replace_sources]
    merged = preserved + incoming
    merged.sort(key=lambda item: (str(item.get("source_file", "")), int(item.get("chunk_order", 0)) if "chunk_order" in item else 0))
    return merged


def selected_source_files(input_dir: Path, source_file: str | None, limit: int | None) -> list[Path]:
    files = count_supported_raw_files(input_dir)
    if source_file:
        files = [path for path in files if source_file in path.name]
    if limit is not None:
        files = files[:limit]
    return files


def print_run_selection(args: argparse.Namespace) -> list[Path]:
    total_raw_files = count_supported_raw_files(args.input_dir)
    selected_files = selected_source_files(args.input_dir, args.source_file, args.limit)

    print("[selection]")
    print(f"  total_raw_files={len(total_raw_files)}")
    print(f"  selected_files={len(selected_files)}")
    print(f"  limit={args.limit if args.limit is not None else 'None'}")
    print(f"  source_file_filter={args.source_file if args.source_file else 'None'}")
    print(f"  selected_file_preview={preview_source_names(selected_files)}")
    print()
    return selected_files


def print_stage_start(stage: str, input_records: int, unique_sources: int) -> None:
    print(f"[{stage}:start]")
    print(f"  input_records={input_records}")
    print(f"  input_unique_sources={unique_sources}")
    print()


def print_stage_end(stage: str, output_records: int, unique_sources: int) -> None:
    print(f"[{stage}:end]")
    print(f"  output_records={output_records}")
    print(f"  output_unique_sources={unique_sources}")
    print()


def run_clean_stage(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_files = selected_source_files(args.input_dir, args.source_file, args.limit)
    if not source_files:
        raise ValueError(f"No source files matched under: {args.input_dir}")

    print_stage_start("clean", input_records=len(source_files), unique_sources=len(source_files))

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for file_path in source_files:
        try:
            original_text = cleaner.read_text_file(file_path)
            cleaned_text = cleaner.clean_text(original_text)
            normalized_units = cleaner.extract_normalized_units(cleaned_text, file_path.name)
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_file": file_path.name, "error": str(exc)})
            continue

        records.append(
            {
                "source_file": file_path.name,
                "relative_path": str(file_path.relative_to(args.input_dir)),
                "title": cleaner.infer_title(file_path),
                "original_text": original_text,
                "cleaned_text": cleaned_text,
                "normalized_units": normalized_units,
            }
        )

    existing = read_json(paths["clean_records"]) if paths["clean_records"].exists() else []
    final_records = merge_by_source_file(existing, records)

    stats = {
        "stage": "clean",
        "selected_sources": len(source_files),
        "cleaned_sources": len(records),
        "input_records": len(source_files),
        "output_records": len(records),
        "output_unique_sources": unique_source_count(records),
        "failed_sources": len(failures),
        "avg_cleaned_chars": round(mean(len(str(item.get("cleaned_text", ""))) for item in records), 2) if records else 0,
        "total_normalized_units": sum(len(item.get("normalized_units", [])) for item in records),
        "normalized_units_needing_manual_review": sum(
            1
            for item in records
            for unit in item.get("normalized_units", [])
            if isinstance(unit, dict) and bool(unit.get("needs_manual_review"))
        ),
        "failures": failures,
    }

    write_json(paths["clean_records"], final_records)
    write_json(paths["clean_stats"], stats)
    print_stage_end("clean", output_records=len(records), unique_sources=unique_source_count(records))
    return final_records, stats


def run_chunk_stage(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths["clean_records"].exists():
        raise FileNotFoundError("Missing cleaned records. Run clean stage first.")

    cleaned_records = apply_filters(read_json(paths["clean_records"]), args.source_file, args.limit)
    if not cleaned_records:
        raise ValueError("No cleaned records matched the current filters.")

    print_stage_start("chunk", input_records=len(cleaned_records), unique_sources=unique_source_count(cleaned_records))

    chunk_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in cleaned_records:
        try:
            chunk_records.extend(
                chunker.build_records_for_sample(
                    sample=record,
                    min_chars=args.min_chars,
                    target_chars=args.target_chars,
                    max_chars=args.max_chars,
                    min_sentences=args.min_sentences,
                    max_sentences=args.max_sentences,
                    overlap_ratio=args.overlap_ratio,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_file": str(record.get("source_file", "")), "error": str(exc)})

    existing = read_json(paths["chunk_records"]) if paths["chunk_records"].exists() else []
    final_records = merge_by_source_file(existing, chunk_records)

    selected_sources = {str(record.get("source_file", "")) for record in cleaned_records}
    stats = {
        "stage": "chunk",
        "selected_sources": len(selected_sources),
        "input_records": len(cleaned_records),
        "output_records": len(chunk_records),
        "output_unique_sources": unique_source_count(chunk_records),
        "total_chunks": len(chunk_records),
        "avg_chunks_per_source": round(len(chunk_records) / max(len(selected_sources), 1), 2),
        "chunks_needing_manual_review": sum(1 for item in chunk_records if bool(item.get("needs_manual_review"))),
        "manual_review_ratio": round(
            sum(1 for item in chunk_records if bool(item.get("needs_manual_review"))) / max(len(chunk_records), 1),
            4,
        ),
        "avg_chunk_chars": round(mean(int(item.get("approx_char_count", 0)) for item in chunk_records), 2) if chunk_records else 0,
        "failures": failures,
    }

    write_json(paths["chunk_records"], final_records)
    write_json(paths["chunk_stats"], stats)
    print_stage_end("chunk", output_records=len(chunk_records), unique_sources=unique_source_count(chunk_records))
    return final_records, stats


def run_insight_stage(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths["chunk_records"].exists():
        raise FileNotFoundError("Missing chunk records. Run chunk stage first.")

    chunk_records = apply_filters(read_json(paths["chunk_records"]), args.source_file, args.limit)
    if not chunk_records:
        raise ValueError("No chunk records matched the current filters.")

    print_stage_start("insight", input_records=len(chunk_records), unique_sources=unique_source_count(chunk_records))

    insight_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for chunk in chunk_records:
        try:
            insights = insighter.build_output_insights(
                chunk=chunk,
                max_items=max(1, min(3, args.max_insights_per_chunk)),
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_file": str(chunk.get("source_file", "")), "chunk_id": str(chunk.get("chunk_id", "")), "error": str(exc)})
            continue

        insight_records.append(
            {
                "source_file": str(chunk.get("source_file", "")),
                "title": str(chunk.get("title", "")),
                "chunk_id": str(chunk.get("chunk_id", "")),
                "chunk_order": int(chunk.get("chunk_order", 0)),
                "chunk_text": str(chunk.get("chunk_text", "")),
                "insights": insights,
            }
        )

    existing = read_json(paths["insight_records"]) if paths["insight_records"].exists() else []
    final_records = merge_by_source_file(existing, insight_records)

    total_insights = sum(len(record.get("insights", [])) for record in insight_records)
    stats = {
        "stage": "insight",
        "selected_chunks": len(chunk_records),
        "input_records": len(chunk_records),
        "output_records": len(insight_records),
        "output_unique_sources": unique_source_count(insight_records),
        "total_insights": total_insights,
        "avg_insights_per_chunk": round(total_insights / max(len(insight_records), 1), 2),
        "insights_needing_manual_review": sum(
            1
            for record in insight_records
            for insight in record.get("insights", [])
            if isinstance(insight, dict) and bool(insight.get("needs_manual_review"))
        ),
        "manual_review_ratio": round(
            sum(
                1
                for record in insight_records
                for insight in record.get("insights", [])
                if isinstance(insight, dict) and bool(insight.get("needs_manual_review"))
            )
            / max(total_insights, 1),
            4,
        ),
        "failures": failures,
    }

    write_json(paths["insight_records"], final_records)
    write_json(paths["insight_stats"], stats)
    print_stage_end("insight", output_records=len(insight_records), unique_sources=unique_source_count(insight_records))
    return final_records, stats


def run_tag_stage(args: argparse.Namespace, paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths["insight_records"].exists():
        raise FileNotFoundError("Missing insight records. Run insight stage first.")

    insight_records = apply_filters(read_json(paths["insight_records"]), args.source_file, args.limit)
    if not insight_records:
        raise ValueError("No insight records matched the current filters.")

    print_stage_start("tag", input_records=len(insight_records), unique_sources=unique_source_count(insight_records))

    tagged_records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in insight_records:
        try:
            tagged_records.append(tagger.tag_record(record))
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_file": str(record.get("source_file", "")), "chunk_id": str(record.get("chunk_id", "")), "error": str(exc)})

    existing = read_json(paths["tag_records"]) if paths["tag_records"].exists() else []
    final_records = merge_by_source_file(existing, tagged_records)

    missing_topic = sum(1 for record in tagged_records if not record.get("tags", {}).get("topic_tags"))
    missing_explicit = sum(
        1 for record in tagged_records if not record.get("tags", {}).get("concepts", {}).get("explicit")
    )
    stats = {
        "stage": "tag",
        "selected_chunks": len(insight_records),
        "input_records": len(insight_records),
        "output_records": len(tagged_records),
        "output_unique_sources": unique_source_count(tagged_records),
        "tagged_chunks": len(tagged_records),
        "tagged_chunks_needing_manual_review": sum(1 for record in tagged_records if bool(record.get("needs_manual_review"))),
        "manual_review_ratio": round(
            sum(1 for record in tagged_records if bool(record.get("needs_manual_review"))) / max(len(tagged_records), 1),
            4,
        ),
        "missing_topic_tags": missing_topic,
        "missing_topic_ratio": round(missing_topic / max(len(tagged_records), 1), 4),
        "missing_explicit_concepts": missing_explicit,
        "missing_explicit_concepts_ratio": round(missing_explicit / max(len(tagged_records), 1), 4),
        "failures": failures,
    }

    write_json(paths["tag_records"], final_records)
    write_json(paths["tag_stats"], stats)
    print_stage_end("tag", output_records=len(tagged_records), unique_sources=unique_source_count(tagged_records))
    return final_records, stats


def top_distribution(values: list[str], limit: int = 10) -> list[dict[str, Any]]:
    counter = Counter(value for value in values if value)
    return [{"label": label, "count": count} for label, count in counter.most_common(limit)]


def validate_pipeline_coverage(paths: dict[str, Path], input_dir: Path) -> dict[str, Any]:
    total_raw_files = len(count_supported_raw_files(input_dir))
    cleaned = read_json(paths["clean_records"]) if paths["clean_records"].exists() else []
    chunks = read_json(paths["chunk_records"]) if paths["chunk_records"].exists() else []
    insights = read_json(paths["insight_records"]) if paths["insight_records"].exists() else []
    tagged = read_json(paths["tag_records"]) if paths["tag_records"].exists() else []

    coverage = {
        "total_raw_files": total_raw_files,
        "cleaned_unique_sources": unique_source_count(cleaned),
        "chunks_unique_sources": unique_source_count(chunks),
        "insights_unique_sources": unique_source_count(insights),
        "tagged_unique_sources": unique_source_count(tagged),
    }
    coverage["coverage_ok"] = (
        coverage["cleaned_unique_sources"] == total_raw_files
        and coverage["chunks_unique_sources"] == total_raw_files
        and coverage["insights_unique_sources"] == total_raw_files
        and coverage["tagged_unique_sources"] == total_raw_files
    )
    return coverage


def build_summary(paths: dict[str, Path], input_dir: Path) -> dict[str, Any]:
    cleaned = read_json(paths["clean_records"]) if paths["clean_records"].exists() else []
    chunks = read_json(paths["chunk_records"]) if paths["chunk_records"].exists() else []
    insights = read_json(paths["insight_records"]) if paths["insight_records"].exists() else []
    tagged = read_json(paths["tag_records"]) if paths["tag_records"].exists() else []
    coverage = validate_pipeline_coverage(paths, input_dir)

    total_insights = sum(len(record.get("insights", [])) for record in insights)
    chunks_needing_manual_review = sum(1 for item in chunks if bool(item.get("needs_manual_review")))
    insights_needing_manual_review = sum(
        1
        for record in insights
        for insight in record.get("insights", [])
        if isinstance(insight, dict) and bool(insight.get("needs_manual_review"))
    )
    tagged_chunks_needing_manual_review = sum(1 for record in tagged if bool(record.get("needs_manual_review")))

    topic_values = [tag for record in tagged for tag in record.get("tags", {}).get("topic_tags", [])]
    scenario_values = [tag for record in tagged for tag in record.get("tags", {}).get("scenario_tags", [])]
    concept_values = [
        concept
        for record in tagged
        for concept in record.get("tags", {}).get("concepts", {}).get("explicit", [])
    ]

    summary = {
        "total_sources": len(cleaned),
        "cleaned_sources": len(cleaned),
        "total_chunks": len(chunks),
        "avg_chunks_per_source": round(len(chunks) / max(len(cleaned), 1), 2),
        "total_insights": total_insights,
        "avg_insights_per_chunk": round(total_insights / max(len(insights), 1), 2),
        "chunks_needing_manual_review": chunks_needing_manual_review,
        "chunks_needing_manual_review_ratio": round(chunks_needing_manual_review / max(len(chunks), 1), 4),
        "insights_needing_manual_review": insights_needing_manual_review,
        "insights_needing_manual_review_ratio": round(insights_needing_manual_review / max(total_insights, 1), 4),
        "tagged_chunks_needing_manual_review": tagged_chunks_needing_manual_review,
        "tagged_chunks_needing_manual_review_ratio": round(tagged_chunks_needing_manual_review / max(len(tagged), 1), 4),
        "missing_topic_tags": sum(1 for record in tagged if not record.get("tags", {}).get("topic_tags")),
        "missing_topic_tags_ratio": round(
            sum(1 for record in tagged if not record.get("tags", {}).get("topic_tags")) / max(len(tagged), 1),
            4,
        ),
        "missing_explicit_concepts": sum(
            1 for record in tagged if not record.get("tags", {}).get("concepts", {}).get("explicit")
        ),
        "missing_explicit_concepts_ratio": round(
            sum(1 for record in tagged if not record.get("tags", {}).get("concepts", {}).get("explicit")) / max(len(tagged), 1),
            4,
        ),
        "top_topics_distribution": top_distribution(topic_values),
        "top_scenarios_distribution": top_distribution(scenario_values),
        "top_concepts_distribution": top_distribution(concept_values),
        "total_raw_files": coverage["total_raw_files"],
        "cleaned_unique_sources": coverage["cleaned_unique_sources"],
        "chunks_unique_sources": coverage["chunks_unique_sources"],
        "insights_unique_sources": coverage["insights_unique_sources"],
        "tagged_unique_sources": coverage["tagged_unique_sources"],
        "coverage_ok": coverage["coverage_ok"],
        "stage_stats": {
            "clean": read_json(paths["clean_stats"]) if paths["clean_stats"].exists() else {},
            "chunk": read_json(paths["chunk_stats"]) if paths["chunk_stats"].exists() else {},
            "insight": read_json(paths["insight_stats"]) if paths["insight_stats"].exists() else {},
            "tag": read_json(paths["tag_stats"]) if paths["tag_stats"].exists() else {},
        },
    }

    write_json(paths["summary"], summary)
    return summary


def print_stage_stats(stage: str, stats: dict[str, Any]) -> None:
    print(f"[{stage}]")
    for key, value in stats.items():
        if key == "failures":
            print(f"  failures: {len(value)}")
        else:
            print(f"  {key}: {value}")
    print()


def print_coverage_status(coverage: dict[str, Any]) -> None:
    print("[coverage]")
    print(f"  total_raw_files={coverage['total_raw_files']}")
    print(f"  cleaned_unique_sources={coverage['cleaned_unique_sources']}")
    print(f"  chunks_unique_sources={coverage['chunks_unique_sources']}")
    print(f"  insights_unique_sources={coverage['insights_unique_sources']}")
    print(f"  tagged_unique_sources={coverage['tagged_unique_sources']}")
    print(f"  coverage_ok={coverage['coverage_ok']}")
    if not coverage["coverage_ok"]:
        print("  WARNING: processed source counts do not fully match data/raw")
    print()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    paths = stage_paths(args.output_root)
    print_run_selection(args)

    if args.stage in {"all", "clean"}:
        _, stats = run_clean_stage(args, paths)
        print_stage_stats("clean", stats)
    if args.stage in {"all", "chunk"}:
        _, stats = run_chunk_stage(args, paths)
        print_stage_stats("chunk", stats)
    if args.stage in {"all", "insight"}:
        _, stats = run_insight_stage(args, paths)
        print_stage_stats("insight", stats)
    if args.stage in {"all", "tag"}:
        _, stats = run_tag_stage(args, paths)
        print_stage_stats("tag", stats)

    if args.stage in {"all", "summary"}:
        summary = build_summary(paths, args.input_dir)
        print("[summary]")
        for key in (
            "total_sources",
            "cleaned_sources",
            "total_chunks",
            "avg_chunks_per_source",
            "total_insights",
            "avg_insights_per_chunk",
            "chunks_needing_manual_review_ratio",
            "insights_needing_manual_review_ratio",
            "tagged_chunks_needing_manual_review_ratio",
            "missing_topic_tags_ratio",
            "missing_explicit_concepts_ratio",
        ):
            print(f"  {key}: {summary[key]}")
        print()
        coverage = {
            "total_raw_files": summary["total_raw_files"],
            "cleaned_unique_sources": summary["cleaned_unique_sources"],
            "chunks_unique_sources": summary["chunks_unique_sources"],
            "insights_unique_sources": summary["insights_unique_sources"],
            "tagged_unique_sources": summary["tagged_unique_sources"],
            "coverage_ok": summary["coverage_ok"],
        }
        print_coverage_status(coverage)


if __name__ == "__main__":
    main()
