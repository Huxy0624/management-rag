#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.api.models.Collection import Collection

from embedding_provider import (
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_LOCAL_MODEL,
    EmbeddingConfigError,
    EmbeddingRuntimeError,
    embed_texts,
    validate_embedding_provider,
)


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_DB_DIR = Path("db/chroma")
DEFAULT_COLLECTION = "management_rag"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_BATCH_SIZE = 100
SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass
class ChunkRecord:
    record_id: str
    document: str
    metadata: dict[str, object]


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip markdown formatting while preserving readable text.
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
    current_length = 0
    search_pos = 0

    def flush_current() -> None:
        nonlocal current_parts, current_length, search_pos
        if not current_parts:
            return
        chunk_text = "\n\n".join(current_parts).strip()
        if not chunk_text:
            current_parts = []
            current_length = 0
            return
        start = text.find(chunk_text, search_pos)
        if start == -1:
            start = search_pos
        end = start + len(chunk_text)
        chunks.append((chunk_text, start, end))
        search_pos = max(start, end - overlap)
        current_parts = []
        current_length = 0

    for paragraph in paragraphs:
        paragraph_length = len(paragraph)

        if paragraph_length > chunk_size:
            flush_current()
            long_parts = split_long_text(paragraph, chunk_size, overlap)
            paragraph_start = text.find(paragraph, search_pos)
            if paragraph_start == -1:
                paragraph_start = search_pos
            for part_text, rel_start, rel_end in long_parts:
                chunks.append(
                    (
                        part_text,
                        paragraph_start + rel_start,
                        paragraph_start + rel_end,
                    )
                )
            search_pos = max(paragraph_start, paragraph_start + paragraph_length - overlap)
            continue

        next_length = current_length + paragraph_length + (2 if current_parts else 0)
        if current_parts and next_length > chunk_size:
            flush_current()

        current_parts.append(paragraph)
        current_length = sum(len(part) for part in current_parts) + max(len(current_parts) - 1, 0) * 2

    flush_current()
    return chunks


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


def build_records(input_dir: Path, chunk_size: int, overlap: int) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []

    for file_path in iter_source_files(input_dir):
        raw_text = read_text_file(file_path)
        cleaned = clean_text(raw_text)
        chunks = split_text(cleaned, chunk_size, overlap)
        if not chunks:
            continue

        source = file_path.name
        title = infer_title(file_path)
        relative_path = file_path.relative_to(input_dir.parent).as_posix()

        for chunk_id, (chunk_text, start, end) in enumerate(chunks):
            record_id = f"{source}:{chunk_id}"
            metadata = {
                "source": source,
                "chunk_id": chunk_id,
                "title": title,
                "relative_path": relative_path,
                "char_start": start,
                "char_end": end,
            }
            records.append(ChunkRecord(record_id=record_id, document=chunk_text, metadata=metadata))

    return records


def chunked(items: list[ChunkRecord], batch_size: int) -> Iterable[list[ChunkRecord]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def delete_existing_sources(collection: Collection, sources: set[str]) -> None:
    for source in sorted(sources):
        collection.delete(where={"source": source})


def embed_and_store(
    records: list[ChunkRecord],
    db_dir: Path,
    collection_name: str,
    embedding_provider: str,
    embedding_model: str,
    batch_size: int,
    reset: bool,
) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(db_dir))

    if reset:
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(name=collection_name)

    if not reset:
        delete_existing_sources(collection, {record.metadata["source"] for record in records})

    total = len(records)
    written = 0

    for batch in chunked(records, batch_size):
        documents = [record.document for record in batch]
        embeddings = embed_texts(
            texts=documents,
            provider=embedding_provider,
            model_name=embedding_model,
        )

        collection.upsert(
            ids=[record.record_id for record in batch],
            documents=documents,
            metadatas=[record.metadata for record in batch],
            embeddings=embeddings,
        )

        written += len(batch)
        print(f"Stored {written}/{total} chunks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local Chroma vector DB from markdown and txt files.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input folder containing .md or .txt files.")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="Persistent Chroma database directory.")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size in characters.")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Character overlap between neighboring chunks.")
    parser.add_argument(
        "--embedding-provider",
        default=DEFAULT_EMBEDDING_PROVIDER,
        choices=["local", "openai"],
        help="Embedding provider. Default is local.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help=f"Embedding model name. Local default: {DEFAULT_LOCAL_MODEL}",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    parser.add_argument("--reset", action="store_true", help="Delete the existing collection before importing.")
    return parser.parse_args()


def validate_environment(input_dir: Path, chunk_size: int, overlap: int, embedding_provider: str) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")

    if overlap < 0:
        raise ValueError("--chunk-overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")

    validate_embedding_provider(embedding_provider)


def run() -> None:
    args = parse_args()
    validate_environment(args.input_dir, args.chunk_size, args.chunk_overlap, args.embedding_provider)

    records = build_records(args.input_dir, args.chunk_size, args.chunk_overlap)
    if not records:
        raise ValueError(f"No .md or .txt files found under: {args.input_dir}")

    unique_sources = sorted({record.metadata["source"] for record in records})
    print(f"Found {len(unique_sources)} files and {len(records)} chunks")

    embed_and_store(
        records=records,
        db_dir=args.db_dir,
        collection_name=args.collection,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        reset=args.reset,
    )
    print(f"Done. Chroma DB saved to: {args.db_dir}")


def main() -> None:
    try:
        run()
    except (EmbeddingConfigError, EmbeddingRuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
