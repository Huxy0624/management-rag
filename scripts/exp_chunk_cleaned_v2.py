#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_PATH_V2 = Path("data/experiments/cleaning_v2/sample_cleaned.json")
DEFAULT_INPUT_PATH_V1 = Path("data/experiments/cleaning_v1/sample_cleaned.json")
DEFAULT_OUTPUT_PATH = Path("data/experiments/chunking_v2/sample_chunks.json")
DEFAULT_LIMIT = 5
DEFAULT_MIN_CHARS = 200
DEFAULT_TARGET_CHARS = 360
DEFAULT_MAX_CHARS = 500
DEFAULT_MIN_SENTENCES = 3
DEFAULT_MAX_SENTENCES = 8
DEFAULT_OVERLAP_RATIO = 0.15


@dataclass
class Segment:
    text: str
    start: int
    end: int
    has_heading_prefix: bool = False


@dataclass
class ChunkDraft:
    start: int
    end: int
    text: str
    has_heading_prefix: bool = False


HEADING_PATTERNS = (
    r"^[一二三四五六七八九十]+、.+",
    r"^第[一二三四五六七八九十0-9]+[章节部分].+",
    r"^[0-9]+[.、].+",
)

NEW_TOPIC_PREFIXES = (
    "一、",
    "二、",
    "三、",
    "四、",
    "五、",
    "六、",
    "七、",
    "八、",
    "九、",
    "十、",
    "另外",
    "此外",
    "同时",
    "另一方面",
    "接下来",
    "结语",
    "总结",
    "综上",
    "因此",
    "所以",
    "方法",
    "案例",
    "定义",
)

TOPIC_TERMS = [
    "信息失真",
    "评价失效",
    "人治",
    "法治",
    "文化建设",
    "机制建设",
    "上升通道",
    "影响力",
    "任务优先级",
    "有效工作",
    "无效工作",
    "不确定性",
    "人效",
    "目标",
    "资源",
    "成本",
    "复盘",
    "专业壁垒",
    "任务",
    "管理",
]


def resolve_input_path(input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path
    if DEFAULT_INPUT_PATH_V2.exists():
        return DEFAULT_INPUT_PATH_V2
    if DEFAULT_INPUT_PATH_V1.exists():
        return DEFAULT_INPUT_PATH_V1
    raise FileNotFoundError(
        f"Neither {DEFAULT_INPUT_PATH_V2} nor {DEFAULT_INPUT_PATH_V1} exists."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated chunking experiments for cleaned knowledge text.")
    parser.add_argument("--input-path", type=Path, help="Input JSON path. Defaults to cleaning_v2, then falls back to cleaning_v1.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON path for chunk samples.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="How many cleaned articles to process.")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help="Preferred minimum chunk size in characters.")
    parser.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS, help="Preferred target chunk size in characters.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS, help="Hard-ish upper bound for chunk size before fallback splitting.")
    parser.add_argument("--min-sentences", type=int, default=DEFAULT_MIN_SENTENCES, help="Preferred minimum sentence count.")
    parser.add_argument("--max-sentences", type=int, default=DEFAULT_MAX_SENTENCES, help="Preferred maximum sentence count.")
    parser.add_argument("--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO, help="Light overlap ratio between neighboring chunks.")
    return parser.parse_args()


def split_paragraphs_with_spans(text: str) -> list[Segment]:
    paragraphs = [part for part in re.split(r"\n{2,}", text) if part.strip()]
    segments: list[Segment] = []
    search_pos = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        start = text.find(paragraph, search_pos)
        if start == -1:
            start = search_pos
        end = start + len(paragraph)
        segments.append(Segment(text=paragraph, start=start, end=end))
        search_pos = end

    return segments


def split_sentences_with_spans(text: str, base_start: int = 0) -> list[Segment]:
    segments: list[Segment] = []
    cursor = 0
    separators = "。！？；"

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.strip()
        if not line:
            cursor += len(raw_line)
            continue

        line_start_in_raw = raw_line.find(line)
        if line_start_in_raw == -1:
            line_start_in_raw = 0
        line_global_start = base_start + cursor + line_start_in_raw

        if is_section_heading_line(line) or (is_list_like(line) and not re.search(r"[。！？；]", line)):
            segments.append(
                Segment(
                    text=line,
                    start=line_global_start,
                    end=line_global_start + len(line),
                )
            )
            cursor += len(raw_line)
            continue

        sentence_start = 0
        for index, char in enumerate(line):
            if char in separators:
                sentence = line[sentence_start : index + 1].strip()
                if sentence:
                    local_start = line.find(sentence, sentence_start, index + 1)
                    if local_start == -1:
                        local_start = sentence_start
                    local_end = local_start + len(sentence)
                    segments.append(
                        Segment(
                            text=sentence,
                            start=line_global_start + local_start,
                            end=line_global_start + local_end,
                        )
                    )
                sentence_start = index + 1

        tail = line[sentence_start:].strip()
        if tail:
            local_start = line.find(tail, sentence_start)
            if local_start == -1:
                local_start = sentence_start
            local_end = local_start + len(tail)
            segments.append(
                Segment(
                    text=tail,
                    start=line_global_start + local_start,
                    end=line_global_start + local_end,
                )
            )

        cursor += len(raw_line)

    return segments


def is_heading_like(paragraph: str) -> bool:
    compact = paragraph.strip()
    if not compact:
        return False
    if any(re.match(pattern, compact) for pattern in HEADING_PATTERNS):
        return True
    if len(compact) <= 20 and not re.search(r"[。！？；：]", compact):
        return True
    return False


def is_section_heading_line(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return False
    if is_list_like(compact):
        return False
    return is_heading_like(compact)


def is_intro_paragraph(paragraph: str) -> bool:
    compact = paragraph.strip()
    return (
        compact.endswith(("：", ":"))
        or "如下案例" in compact
        or "举个例子" in compact
        or "比如" in compact
        or "可分为" in compact
    )


def is_list_like(paragraph: str) -> bool:
    return bool(
        re.match(r"^[0-9]+[.、]", paragraph.strip())
        or re.search(r"\n\s*[0-9]+[.、]", paragraph)
        or paragraph.strip().startswith(">")
    )


def build_semantic_units(cleaned_text: str) -> list[Segment]:
    paragraphs = split_paragraphs_with_spans(cleaned_text)
    units: list[Segment] = []
    pending_headings: list[Segment] = []
    index = 0

    while index < len(paragraphs):
        paragraph = paragraphs[index]

        if is_heading_like(paragraph.text):
            pending_headings.append(paragraph)
            index += 1
            continue

        start = pending_headings[0].start if pending_headings else paragraph.start
        end = paragraph.end
        unit_parts = [heading.text for heading in pending_headings] + [paragraph.text]
        has_heading_prefix = bool(pending_headings)
        pending_headings = []

        if is_intro_paragraph(paragraph.text):
            next_index = index + 1
            while next_index < len(paragraphs):
                next_paragraph = paragraphs[next_index]
                if is_heading_like(next_paragraph.text):
                    break
                if is_list_like(next_paragraph.text) or len(next_paragraph.text) <= 140:
                    unit_parts.append(next_paragraph.text)
                    end = next_paragraph.end
                    next_index += 1
                    continue
                break
            index = next_index
        else:
            index += 1

        units.append(
            Segment(
                text="\n\n".join(unit_parts).strip(),
                start=start,
                end=end,
                has_heading_prefix=has_heading_prefix,
            )
        )

    for heading in pending_headings:
        units.append(
            Segment(
                text=heading.text,
                start=heading.start,
                end=heading.end,
                has_heading_prefix=True,
            )
        )

    return units


def starts_new_topic(sentence: str) -> bool:
    compact = sentence.strip()
    return any(compact.startswith(prefix) for prefix in NEW_TOPIC_PREFIXES)


def split_long_unit(
    unit: Segment,
    target_chars: int,
    max_chars: int,
    min_sentences: int,
    max_sentences: int,
) -> list[Segment]:
    sentences = split_sentences_with_spans(unit.text, unit.start)
    if not sentences:
        return [unit]

    chunks: list[Segment] = []
    current_sentences: list[Segment] = []

    def flush() -> None:
        nonlocal current_sentences
        if not current_sentences:
            return
        chunk_start = current_sentences[0].start
        chunk_end = current_sentences[-1].end
        chunk_text = " ".join(sentence.text.strip() for sentence in current_sentences if sentence.text.strip())
        chunks.append(
            Segment(
                text=chunk_text.strip(),
                start=chunk_start,
                end=chunk_end,
                has_heading_prefix=unit.has_heading_prefix and len(chunks) == 0,
            )
        )
        current_sentences = []

    for sentence in sentences:
        candidate = current_sentences + [sentence]
        candidate_text = " ".join(item.text.strip() for item in candidate if item.text.strip())
        candidate_chars = len(candidate_text)

        if not current_sentences:
            current_sentences.append(sentence)
            continue

        should_flush = False
        if candidate_chars > max_chars and len(current_sentences) >= min_sentences:
            should_flush = True
        elif len(current_sentences) >= max_sentences:
            should_flush = True
        elif len(current_sentences) >= min_sentences and len(candidate_text) >= target_chars and starts_new_topic(sentence.text):
            should_flush = True

        if should_flush:
            flush()
            current_sentences.append(sentence)
        else:
            current_sentences.append(sentence)

    flush()
    return chunks


def refine_units(
    units: list[Segment],
    target_chars: int,
    max_chars: int,
    min_sentences: int,
    max_sentences: int,
) -> list[Segment]:
    refined: list[Segment] = []
    for unit in units:
        sentence_count = len(split_sentences_with_spans(unit.text))
        if len(unit.text) > max_chars or sentence_count > max_sentences:
            refined.extend(split_long_unit(unit, target_chars, max_chars, min_sentences, max_sentences))
        else:
            refined.append(unit)
    return refined


def text_topics(text: str) -> set[str]:
    return {term for term in TOPIC_TERMS if term in text}


def likely_cross_topic(current_text: str, next_text: str) -> bool:
    current_topics = text_topics(current_text)
    next_topics = text_topics(next_text)
    if not current_topics or not next_topics:
        return False
    if current_topics & next_topics:
        return False
    return len(current_topics | next_topics) >= 3


def build_chunk_drafts(
    cleaned_text: str,
    units: list[Segment],
    min_chars: int,
    target_chars: int,
    max_chars: int,
    min_sentences: int,
    max_sentences: int,
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    current_units: list[Segment] = []

    def flush() -> None:
        nonlocal current_units
        if not current_units:
            return
        start = current_units[0].start
        end = current_units[-1].end
        drafts.append(
            ChunkDraft(
                start=start,
                end=end,
                text=cleaned_text[start:end].strip(),
                has_heading_prefix=current_units[0].has_heading_prefix,
            )
        )
        current_units = []

    for unit in units:
        if not current_units:
            current_units.append(unit)
            continue

        current_text = cleaned_text[current_units[0].start : current_units[-1].end].strip()
        candidate_text = cleaned_text[current_units[0].start : unit.end].strip()
        candidate_chars = len(candidate_text)
        candidate_sentences = len(split_sentences_with_spans(candidate_text))

        hard_boundary = unit.has_heading_prefix
        cross_topic = likely_cross_topic(current_text, unit.text)

        should_flush = False
        if candidate_chars > max_chars and len(current_text) >= min_chars:
            should_flush = True
        elif candidate_sentences > max_sentences and len(current_text) >= min_chars:
            should_flush = True
        elif hard_boundary and len(current_text) >= max(min_chars // 2, 120):
            should_flush = True
        elif cross_topic and len(current_text) >= max(min_chars // 2, 120):
            should_flush = True
        elif len(current_text) >= target_chars and starts_new_topic(unit.text):
            should_flush = True

        if should_flush:
            flush()
            current_units.append(unit)
        else:
            current_units.append(unit)

    flush()
    return drafts


def local_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    return [(segment.start, segment.end, segment.text) for segment in split_sentences_with_spans(text, 0)]


def tail_overlap_text(text: str, overlap_ratio: float) -> tuple[str, int]:
    target_chars = max(1, int(len(text) * overlap_ratio))
    spans = local_sentence_spans(text)
    if not spans:
        overlap_text = text[-target_chars:].strip()
        overlap_start = max(0, len(text) - len(overlap_text))
        return overlap_text, overlap_start

    selected: list[tuple[int, int, str]] = []
    total = 0
    for span in reversed(spans):
        selected.append(span)
        total += len(span[2])
        if total >= target_chars:
            break
    selected.reverse()
    overlap_start = selected[0][0]
    overlap_text = text[overlap_start : selected[-1][1]].strip()
    return overlap_text, overlap_start


def apply_overlap(
    drafts: list[ChunkDraft],
    overlap_ratio: float,
) -> list[dict[str, object]]:
    outputs: list[dict[str, object]] = []

    for index, draft in enumerate(drafts):
        overlap_text = ""
        overlap_span = ""
        chunk_text = draft.text
        span_start = draft.start

        if index > 0 and overlap_ratio > 0:
            previous = drafts[index - 1]
            overlap_text, local_overlap_start = tail_overlap_text(previous.text, overlap_ratio)
            if overlap_text:
                overlap_start = previous.start + local_overlap_start
                overlap_span = f", overlap:{overlap_start}-{previous.end}"
                span_start = overlap_start
                chunk_text = f"{overlap_text}\n\n{draft.text}".strip()

        outputs.append(
            {
                "chunk_text": chunk_text,
                "span_start": span_start,
                "span_end": draft.end,
                "overlap_span": overlap_span,
            }
        )

    return outputs


def chunk_needs_manual_review(
    chunk_text: str,
    min_chars: int,
    max_chars: int,
    min_sentences: int,
    max_sentences: int,
) -> bool:
    char_count = len(chunk_text)
    sentence_count = len(split_sentences_with_spans(chunk_text))
    topic_count = len(text_topics(chunk_text))
    heading_count = sum(1 for line in chunk_text.splitlines() if is_section_heading_line(line.strip()))

    if char_count < max(120, min_chars // 2):
        return True
    if char_count > max_chars + 120:
        return True
    if sentence_count < max(1, min_sentences - 2):
        return True
    if sentence_count > max_sentences + 5 and char_count > max_chars:
        return True
    if heading_count > 2:
        return True
    if topic_count >= 6 and char_count > max_chars:
        return True
    return False


def build_records_for_sample(
    sample: dict[str, object],
    min_chars: int,
    target_chars: int,
    max_chars: int,
    min_sentences: int,
    max_sentences: int,
    overlap_ratio: float,
) -> list[dict[str, object]]:
    cleaned_text = str(sample.get("cleaned_text", "")).strip()
    if not cleaned_text:
        return []

    semantic_units = build_semantic_units(cleaned_text)
    refined_units = refine_units(
        units=semantic_units,
        target_chars=target_chars,
        max_chars=max_chars,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )
    chunk_drafts = build_chunk_drafts(
        cleaned_text=cleaned_text,
        units=refined_units,
        min_chars=min_chars,
        target_chars=target_chars,
        max_chars=max_chars,
        min_sentences=min_sentences,
        max_sentences=max_sentences,
    )
    overlapped = apply_overlap(chunk_drafts, overlap_ratio)

    records: list[dict[str, object]] = []
    for order, (draft, overlap_info) in enumerate(zip(chunk_drafts, overlapped)):
        chunk_text = str(overlap_info["chunk_text"])
        approx_char_count = len(chunk_text)
        approx_sentence_count = len(split_sentences_with_spans(chunk_text))
        source_span = f"chars:{draft.start}-{draft.end}{overlap_info['overlap_span']}"

        records.append(
            {
                "source_file": str(sample.get("source_file", "")),
                "title": str(sample.get("title", "")),
                "chunk_id": f"{sample.get('source_file', '')}:{order}",
                "chunk_order": order,
                "source_span": source_span,
                "chunk_text": chunk_text,
                "approx_char_count": approx_char_count,
                "approx_sentence_count": approx_sentence_count,
                "needs_manual_review": chunk_needs_manual_review(
                    chunk_text,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    min_sentences=min_sentences,
                    max_sentences=max_sentences,
                ),
            }
        )

    return records


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if args.min_chars <= 0 or args.target_chars <= 0 or args.max_chars <= 0:
        raise ValueError("chunk character limits must be greater than 0")
    if not (0 <= args.overlap_ratio < 1):
        raise ValueError("--overlap-ratio must be in [0, 1)")
    if not (args.min_chars <= args.target_chars <= args.max_chars):
        raise ValueError("Expected min_chars <= target_chars <= max_chars")

    cleaned_samples = json.loads(input_path.read_text(encoding="utf-8"))[: args.limit]
    chunk_records: list[dict[str, object]] = []

    for sample in cleaned_samples:
        chunk_records.extend(
            build_records_for_sample(
                sample=sample,
                min_chars=args.min_chars,
                target_chars=args.target_chars,
                max_chars=args.max_chars,
                min_sentences=args.min_sentences,
                max_sentences=args.max_sentences,
                overlap_ratio=args.overlap_ratio,
            )
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(chunk_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input file: {input_path}")
    print(f"Processed {len(cleaned_samples)} cleaned articles")
    print(f"Generated {len(chunk_records)} chunks")
    print(f"Saved chunk samples to: {args.output_path}")


if __name__ == "__main__":
    main()
