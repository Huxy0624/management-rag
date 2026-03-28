#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/experiments/cleaning_v2/sample_cleaned.json")
SUPPORTED_EXTENSIONS = {".md", ".txt"}

NOISE_LINE_PATTERNS = [
    r"关注公众号.*",
    r".*回复\s*\d+.*",
    r".*加我微信.*",
    r".*私信.*",
    r".*点赞.*",
    r".*收藏.*",
    r".*转发.*",
    r".*记得点赞关注.*",
    r".*这是管理课程.*",
    r".*下期再讲.*",
    r".*我们继续上一节.*",
    r".*刚刚讲到哪里.*",
    r".*下一节我们会继续.*",
]

NOISE_SENTENCE_PATTERNS = [
    r"^关注公众号.*",
    r"^回复\s*\d+.*",
    r"^加我微信.*",
    r"^记得点赞关注.*",
    r"^请点赞.*",
    r"^请收藏.*",
    r"^请转发.*",
    r"^下期再讲.*",
    r"^好了[，,].*下一节.*",
    r"^我们今天一起探讨了.*下一节.*",
    r"^在前两节内容的基础上[，,].*今天的内容是[:：]?$",
    r"^一些同学可能会认为他们.*$",
    r"^在这个时候你会不会.*$",
    r"^情况如图所示[:：]?$",
    r"^如图所示[:：]?$",
    r"^这是今天内容的重点.*$",
    r"^公司人数规模与有效工作的关系是什么？?$",
    r"^这里先说下文化建设[。]?$",
    r"^文化建设[。]?$",
    r"^我们探讨了管理的目标是什么？?$",
]

CONVERSATIONAL_PREFIX_PATTERNS = [
    (r"^上一节[，,]?", ""),
    (r"^前文回顾[：:，,]?", ""),
    (r"^课程回顾[：:，,]?", ""),
    (r"^在前两节内容的基础上[，,]我们来继续进一步打开管理的本质问题[，,]今天的内容是[:：]?", ""),
    (r"^这里先说下", ""),
    (r"^这里需要回答一个问题[:：]?", ""),
]

EMPHASIS_PREFIX_PATTERNS = [
    (r"^(这个非常重要[，,：:]?)", ""),
    (r"^(这个很重要[，,：:]?)", ""),
    (r"^(这一点非常重要[，,：:]?)", ""),
    (r"^(大家一定要记住[，,：:]?)", ""),
    (r"^(一定要记住[，,：:]?)", ""),
    (r"^(一定要注意[，,：:]?)", ""),
    (r"^(这里需要非常注意的是)", ""),
    (r"^(这里要注意[，,：:]?)", ""),
]

CONCEPT_TERMS = [
    "管理",
    "人效",
    "有效工作",
    "无效工作",
    "信息失真",
    "评价失效",
    "评价权",
    "资源",
    "成本",
    "信息差",
    "文化建设",
    "机制建设",
    "人治",
    "法治",
    "任务优先级",
    "影响力",
    "上升通道",
    "不确定性",
    "专业壁垒",
    "目标",
    "效率",
    "复盘",
    "公司规模",
    "沟通复杂度",
]

COLLOQUIAL_NORMALIZATION_MAP = {
    "垃圾但必须吃的屎粑粑": "低价值但必要的事务性工作",
    "英雄送葬场": "系统性问题向基层管理者集中外溢",
    "绞肉机": "高消耗高风险任务",
    "白嫖": "低成本获取员工额外投入",
    "画大饼": "通过远期承诺激励团队",
}


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


def strip_markdown_and_links(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.M)
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("`", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def is_noise_line(line: str) -> bool:
    compact = line.strip()
    if not compact:
        return True
    if compact in {"---", "***"}:
        return True
    if re.fullmatch(r"https?://\S+", compact):
        return True
    if re.fullmatch(r"第[一二三四五六七八九十0-9]+节[:：].*", compact):
        return True
    if "files.mdnice.com" in compact:
        return True
    return any(re.search(pattern, compact, flags=re.I) for pattern in NOISE_LINE_PATTERNS)


def strip_emphasis_prefix(sentence: str) -> str:
    sentence = sentence.strip()
    for pattern, replacement in EMPHASIS_PREFIX_PATTERNS:
        sentence = re.sub(pattern, replacement, sentence)
    return sentence.strip(" ，,：:")


def is_noise_sentence(sentence: str) -> bool:
    compact = sentence.strip()
    if not compact:
        return True
    if compact in {"...", "……"}:
        return True
    return any(re.search(pattern, compact) for pattern in NOISE_SENTENCE_PATTERNS)


def clean_sentence(sentence: str) -> str:
    cleaned = sentence.strip()
    for pattern, replacement in CONVERSATIONAL_PREFIX_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned)

    cleaned = re.sub(
        r"^其答案是[:：]?",
        "管理的目标是：",
        cleaned,
    )
    cleaned = re.sub(
        r"^今天的内容是[:：]?",
        "",
        cleaned,
    )
    cleaned = strip_emphasis_prefix(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])", paragraph)
    return [part.strip() for part in parts if part.strip()]


def clean_paragraph(paragraph: str) -> str:
    lines = [line.strip() for line in paragraph.splitlines()]
    lines = [line for line in lines if not is_noise_line(line)]
    if not lines:
        return ""

    paragraph_text = "\n".join(lines).strip()
    if not paragraph_text:
        return ""

    if re.fullmatch(r"第[一二三四五六七八九十0-9]+节[:：].*", paragraph_text):
        return ""
    if paragraph_text.startswith("课程回顾"):
        return ""
    if "知识重点为" in paragraph_text:
        return ""

    if "\n" in paragraph_text or re.match(r"^\d+\.", paragraph_text):
        cleaned_lines: list[str] = []
        for line in paragraph_text.splitlines():
            cleaned_line = clean_sentence(line)
            if not cleaned_line:
                continue
            if is_noise_sentence(cleaned_line):
                continue
            cleaned_lines.append(cleaned_line)
        return "\n".join(cleaned_lines).strip()

    sentences = split_sentences(paragraph_text)
    kept_sentences: list[str] = []
    for sentence in sentences:
        cleaned_sentence = clean_sentence(sentence)
        if not cleaned_sentence:
            continue
        if is_noise_sentence(cleaned_sentence):
            continue
        kept_sentences.append(cleaned_sentence)

    return "".join(kept_sentences).strip()


def deduplicate_paragraphs(paragraphs: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for paragraph in paragraphs:
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(paragraph.strip())
    return deduped


def clean_text(text: str) -> str:
    text = strip_markdown_and_links(text)
    raw_paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]

    cleaned_paragraphs: list[str] = []
    for paragraph in raw_paragraphs:
        cleaned = clean_paragraph(paragraph)
        if not cleaned:
            continue
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        if cleaned:
            cleaned_paragraphs.append(cleaned)

    cleaned_paragraphs = deduplicate_paragraphs(cleaned_paragraphs)
    return "\n\n".join(cleaned_paragraphs).strip()


def normalize_unit_text(text: str) -> tuple[str, list[str]]:
    normalized = text.strip()
    inferred_concepts: list[str] = []

    for source_phrase, target_phrase in COLLOQUIAL_NORMALIZATION_MAP.items():
        if source_phrase in normalized:
            normalized = normalized.replace(source_phrase, target_phrase)
            inferred_concepts.append(target_phrase)

    normalized = re.sub(r"^综上[，,:：]?", "", normalized).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized, inferred_concepts


def extract_explicit_concepts(text: str) -> list[str]:
    return [concept for concept in CONCEPT_TERMS if concept in text]


def classify_unit_type(paragraph: str) -> str | None:
    if any(pattern in paragraph for pattern in ("本质是", "定义是", "是指", "意味着")):
        return "definition"
    if any(pattern in paragraph for pattern in ("模型", "方法", "步骤", "解法", "应用方法")):
        return "method"
    if any(pattern in paragraph for pattern in ("案例", "举个例子", "如以下案例")):
        return "case"
    if any(pattern in paragraph for pattern in ("不要", "不能", "避免", "风险", "陷阱")):
        return "warning"
    if any(pattern in paragraph for pattern in ("导致", "决定", "影响", "关系", "根源是")):
        return "relation"
    if any(pattern in paragraph for pattern in ("目标是", "关键在于", "需要", "应该", "核心是")):
        return "principle"
    return None


def extract_normalized_units(cleaned_text: str, source_file: str, max_units: int = 8) -> list[dict[str, object]]:
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", cleaned_text) if part.strip()]
    units: list[dict[str, object]] = []
    search_pos = 0

    for paragraph in paragraphs:
        if len(paragraph) < 25:
            continue
        if any(
            marker in paragraph
            for marker in (
                "这是今天内容的重点",
                "结语",
                "网上也流传着",
                "一些同学可能会认为",
                "在这个时候你会不会",
            )
        ):
            continue

        unit_type = classify_unit_type(paragraph)
        if unit_type is None:
            continue

        start = cleaned_text.find(paragraph, search_pos)
        if start == -1:
            start = search_pos
        end = start + len(paragraph)
        search_pos = end

        normalized_text, inferred_concepts = normalize_unit_text(paragraph)
        explicit_concepts = extract_explicit_concepts(paragraph)
        if not explicit_concepts and unit_type != "case":
            continue

        needs_manual_review = (
            bool(inferred_concepts)
            or unit_type in {"case", "relation", "warning"}
            or len(normalized_text) > 220
            or not explicit_concepts
        )

        unit_id = f"{source_file}:unit:{len(units)}"
        units.append(
            {
                "unit_id": unit_id,
                "unit_type": unit_type,
                "source_span": f"chars:{start}-{end}",
                "normalized_text": normalized_text,
                "explicit_concepts": explicit_concepts,
                "inferred_concepts": inferred_concepts,
                "needs_manual_review": needs_manual_review,
            }
        )

        if len(units) >= max_units:
            break

    return units


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated article cleaning experiments for knowledge-base readiness.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Input folder containing .md or .txt files.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSON path for cleaned samples.")
    parser.add_argument("--limit", type=int, default=3, help="How many source files to process.")
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

    samples: list[dict[str, object]] = []
    for file_path in source_files:
        original_text = read_text_file(file_path)
        cleaned_text = clean_text(original_text)
        normalized_units = extract_normalized_units(cleaned_text, file_path.name)

        samples.append(
            {
                "source_file": file_path.name,
                "title": infer_title(file_path),
                "original_text": original_text,
                "cleaned_text": cleaned_text,
                "normalized_units": normalized_units,
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
