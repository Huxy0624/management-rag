#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/experiments/cleaning_v1/sample_cleaned.json")
DEFAULT_OUTPUT_PATH = Path("data/experiments/chunking_v1/sample_chunks.json")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    parts: list[tuple[str, int, int]] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            parts.append((chunk, start, end))
        if end >= text_length:
            break
        start = max(end - overlap, start + 1)

    return parts


def split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    text = text.strip()
    if not text:
        return []

    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    if not paragraphs:
        return split_long_text(text, chunk_size, overlap)

    chunks: list[tuple[str, int, int]] = []
    current_parts: list[str] = []
    search_pos = 0

    def flush_current() -> None:
        nonlocal current_parts, search_pos
        if not current_parts:
            return
        chunk_text = "\n\n".join(current_parts).strip()
        if not chunk_text:
            current_parts = []
            return
        start = text.find(chunk_text, search_pos)
        if start == -1:
            start = search_pos
        end = start + len(chunk_text)
        chunks.append((chunk_text, start, end))
        search_pos = max(start, end - overlap)
        current_parts = []

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            flush_current()
            paragraph_start = text.find(paragraph, search_pos)
            if paragraph_start == -1:
                paragraph_start = search_pos

            for part_text, rel_start, rel_end in split_long_text(paragraph, chunk_size, overlap):
                chunks.append(
                    (
                        part_text,
                        paragraph_start + rel_start,
                        paragraph_start + rel_end,
                    )
                )
            search_pos = max(paragraph_start, paragraph_start + len(paragraph) - overlap)
            continue

        next_text = "\n\n".join(current_parts + [paragraph]).strip()
        if current_parts and len(next_text) > chunk_size:
            flush_current()

        current_parts.append(paragraph)

    flush_current()
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated chunking experiments on cleaned article samples.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Input JSON path from cleaning experiment.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON path for chunk samples.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters.")
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap in characters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")
    if args.overlap < 0:
        raise ValueError("--overlap cannot be negative")
    if args.overlap >= args.chunk_size:
        raise ValueError("--overlap must be smaller than --chunk-size")

    cleaned_samples = json.loads(args.input_path.read_text(encoding="utf-8"))
    chunk_records: list[dict[str, object]] = []

    for sample in cleaned_samples:
        cleaned_text = str(sample.get("cleaned_text", "")).strip()
        source_file = str(sample.get("source_file", ""))
        title = str(sample.get("title", ""))

        for chunk_order, (chunk_text, start, end) in enumerate(split_text(cleaned_text, args.chunk_size, args.overlap)):
            chunk_records.append(
                {
                    "source_file": source_file,
                    "title": title,
                    "chunk_id": f"{source_file}:{chunk_order}",
                    "chunk_order": chunk_order,
                    "source_span": {
                        "start": start,
                        "end": end,
                    },
                    "chunk_text": chunk_text,
                }
            )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(chunk_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Processed {len(cleaned_samples)} cleaned articles")
    print(f"Generated {len(chunk_records)} chunks")
    print(f"Saved chunk samples to: {args.output_path}")


if __name__ == "__main__":
    main()
