#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/experiments/cleaning_v1/sample_cleaned.json")
SUPPORTED_EXTENSIONS = {".md", ".txt"}


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.M)
    text = re.sub(r"[*_~]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)

    return text.strip()


def read_text_file(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "gb18030", "gbk")
    last_error: UnicodeDecodeError | None = None

    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    if last_error:
        raise last_error
    return path.read_text()


def infer_title(file_path: Path) -> str:
    title = file_path.stem
    title = re.sub(r"[_-]\d{4}-\d{2}-\d{2}$", "", title)
    title = re.sub(r"\s+", " ", title).strip(" _-")
    return title or file_path.stem


def iter_source_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated article cleaning experiments.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input folder containing .md or .txt files.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON path for cleaned samples.")
    parser.add_argument("--limit", type=int, default=5, help="How many source files to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0")

    source_files = list(iter_source_files(args.input_dir))[: args.limit]
    if not source_files:
        raise ValueError(f"No .md or .txt files found under: {args.input_dir}")

    samples: list[dict[str, str]] = []
    for file_path in source_files:
        original_text = read_text_file(file_path)
        cleaned_text = clean_text(original_text)
        samples.append(
            {
                "source_file": file_path.name,
                "title": infer_title(file_path),
                "original_text": original_text,
                "cleaned_text": cleaned_text,
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Processed {len(samples)} files")
    print(f"Saved cleaned samples to: {args.output_path}")


if __name__ == "__main__":
    main()
