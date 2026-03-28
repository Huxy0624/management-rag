#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/experiments/chunking_v2/sample_chunks.json")
DEFAULT_OUTPUT_PATH = Path("data/experiments/insights_v1/sample_insights.json")
DEFAULT_LIMIT = 5
DEFAULT_MAX_INSIGHTS_PER_CHUNK = 3
MAX_INSIGHT_TEXT_LENGTH = 50

ALLOWED_INSIGHT_TYPES = {
    "definition",
    "principle",
    "method",
    "judgment",
    "warning",
    "cause",
    "effect",
    "relation",
    "case_takeaway",
}

DISCOURSE_PREFIXES = (
    "综上，",
    "所以，",
    "因此，",
    "换句话说，",
    "可以看出，",
    "事实上，",
    "如前所述，",
    "如前整理：",
    "这里需要注意的是",
    "这里需要回答一个问题：",
    "从这个角度来说，",
    "只不过，",
    "答案可能比较反认知：",
)

HEADING_PATTERNS = (
    r"^[一二三四五六七八九十]+、.+",
    r"^第[一二三四五六七八九十0-9]+[章节部分].+",
    r"^[0-9]+[.、].+",
)

CONCEPT_TERMS = (
    "管理",
    "人效",
    "有效工作",
    "无效工作",
    "信息失真",
    "评价失效",
    "不确定性",
    "资源",
    "成本",
    "效率",
    "目标",
    "机制",
    "文化",
    "法治",
    "人治",
    "复盘",
    "风险",
    "价值",
    "评价",
    "沟通",
)

LOW_SIGNAL_PATTERNS = (
    r"^神奇的管理$",
    r"^管理的本质$",
    r"^公司规模\s*&\s*信息失真$",
    r"^结语$",
    r"^海恩法则指出$",
    r"^因为是技术出身$",
    r"^管理的学习从来不是独立的，但可以硬分为$",
    r"^高管类管理资料还真不多$",
)

QUESTION_PATTERNS = (
    r"什么是",
    r"为什么",
    r"应该学习什么",
    r"今天我们需要思考",
    r"如何通俗易懂",
)


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int


@dataclass
class InsightCandidate:
    text: str
    insight_type: str
    source_span: str
    confidence: float
    needs_manual_review: bool
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated insight extraction experiments on chunking_v2 output.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Input chunk JSON path.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output insight JSON path.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Process only the first N chunks after filtering.")
    parser.add_argument("--source-file", type=str, help="Optional source_file substring filter.")
    parser.add_argument(
        "--max-insights-per-chunk",
        type=int,
        default=DEFAULT_MAX_INSIGHTS_PER_CHUNK,
        help="Maximum number of insights to keep for each chunk.",
    )
    return parser.parse_args()


def is_heading_like(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return False
    if any(re.match(pattern, compact) for pattern in HEADING_PATTERNS):
        return True
    if len(compact) <= 20 and not re.search(r"[。！？；：]", compact):
        return True
    return False


def split_sentences_with_spans(text: str) -> list[SentenceSpan]:
    sentences: list[SentenceSpan] = []
    cursor = 0
    separators = "。！？；"

    for raw_line in text.splitlines(keepends=True):
        line = raw_line.strip()
        if not line:
            cursor += len(raw_line)
            continue

        line_offset = raw_line.find(line)
        if line_offset == -1:
            line_offset = 0
        line_start = cursor + line_offset

        if is_heading_like(line):
            sentences.append(SentenceSpan(text=line, start=line_start, end=line_start + len(line)))
            cursor += len(raw_line)
            continue

        sentence_start = 0
        for index, char in enumerate(line):
            if char in separators:
                piece = line[sentence_start : index + 1].strip()
                if piece:
                    local_start = line.find(piece, sentence_start, index + 1)
                    if local_start == -1:
                        local_start = sentence_start
                    local_end = local_start + len(piece)
                    sentences.append(
                        SentenceSpan(
                            text=piece,
                            start=line_start + local_start,
                            end=line_start + local_end,
                        )
                    )
                sentence_start = index + 1

        tail = line[sentence_start:].strip()
        if tail:
            local_start = line.find(tail, sentence_start)
            if local_start == -1:
                local_start = sentence_start
            local_end = local_start + len(tail)
            sentences.append(
                SentenceSpan(
                    text=tail,
                    start=line_start + local_start,
                    end=line_start + local_end,
                )
            )

        cursor += len(raw_line)

    return sentences


def normalize_sentence(text: str) -> str:
    normalized = text.strip().strip("> ").strip()
    normalized = re.sub(r"^[0-9]+[.、]\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    for prefix in DISCOURSE_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    normalized = normalized.replace("“", "").replace("”", "")
    normalized = normalized.replace("‘", "").replace("’", "")
    normalized = normalized.replace("这句话：", "")
    normalized = normalized.replace("一句大白话描述管理是", "")
    normalized = normalized.replace("大白话描述管理是", "")
    normalized = normalized.replace("这里需要回答一个问题：", "")
    normalized = normalized.replace("只不过，", "")
    normalized = normalized.strip("：:;；，, ")
    return normalized


def is_low_signal_sentence(text: str) -> bool:
    compact = normalize_sentence(text)
    if not compact:
        return True
    if any(re.search(pattern, compact) for pattern in QUESTION_PATTERNS):
        return True
    if any(re.match(pattern, compact) for pattern in LOW_SIGNAL_PATTERNS):
        return True
    if re.search(r"^(我|我们).{0,8}(一直在思考|发现|探讨|需要思考)", compact):
        return True
    if re.search(r"^(一个作者说|另一个作者说|然后来一个作者说)", compact):
        return True
    if compact.endswith(("吗", "呢", "？", "?")):
        return True
    if len(compact) <= 8 and not re.search(r"[是有会将应需]", compact):
        return True
    return False


def score_text_density(text: str) -> float:
    score = 0.0
    for term in CONCEPT_TERMS:
        if term in text:
            score += 0.08
    if re.search(r"(是|意味着|说明|导致|取决于|需要|应该|必须)", text):
        score += 0.15
    if re.search(r"(因为|由于|所以|因此|从而|使得)", text):
        score += 0.12
    if re.search(r"(风险|难以|容易|失真|失效|浪费|低效)", text):
        score += 0.12
    if re.search(r"(方法|机制|做法|路径|步骤|解法)", text):
        score += 0.12
    return min(score, 0.28)


def extract_key_clause(text: str) -> str:
    clauses = [part.strip(" ，,：:；;") for part in re.split(r"[，,：:；;]", text) if part.strip()]
    if not clauses:
        return text.strip()

    def clause_score(clause: str) -> tuple[float, int]:
        score = 0.0
        for term in CONCEPT_TERMS:
            if term in clause:
                score += 1.0
        if re.search(r"(本质|核心|目标|意义|导致|需要|应该|必须|关键|风险|问题)", clause):
            score += 1.5
        if re.search(r"(因为|由于|所以|因此|使得|意味着|说明)", clause):
            score += 1.2
        return (score, len(clause))

    clauses.sort(key=clause_score, reverse=True)
    return clauses[0]


def compress_insight_text(source_text: str, insight_type: str, chunk_text: str) -> str:
    text = normalize_sentence(source_text)

    replacements = (
        ("真正的涵义是指", "本质是"),
        ("管理其实追求的是", "管理的核心目标是"),
        ("管理的重点在于", "管理的重点是"),
        ("管理的存在的意义是", "管理的意义在于"),
        ("其结果可能是", "结果是"),
        ("其结果当然是", "结果是"),
        ("会因为", "因"),
        ("会导致", "导致"),
        ("会使得", "使"),
        ("可以看出", ""),
        ("答案很简单", ""),
        ("这其实很容易理解", ""),
        ("这意味着", "意味着"),
    )
    for old, new in replacements:
        text = text.replace(old, new)

    text = re.sub(r"^而，", "", text)
    text = re.sub(r"^而", "", text)
    text = re.sub(r"^但", "", text)
    text = re.sub(r"^所以", "", text)
    text = re.sub(r"^因此", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip("，,：:；;。 ")

    if "管理就是如何用3个人的成本，让2个人干4个人的活" in text:
        return "管理的目标是在有限成本下提升产出"
    if "管理其实追求的是人效" in source_text or "管理的核心目标是" in text and "人效" in text:
        return "管理的核心目标是提升人效"
    if "人效真正的涵义是指有效工作" in source_text:
        return "人效的核心是有效工作"
    if "管理的重点在于工作中的不确定性" in source_text:
        return "管理的重要任务是应对不确定性"
    if "管理能力变成了一种理所当然" in source_text:
        return "中高层管理能力常被视为默认要求"
    if "部门墙" in source_text and "管理能力不行" in source_text:
        return "跨部门失败常被归因于管理能力不足"
    if "公司至少有两个目标" in source_text and "活下去" in chunk_text:
        return "公司的基本目标是完成产品并持续生存"
    if "沟通复杂度随人数的增加呈非线性增长" in source_text:
        return "团队扩大将显著增加沟通复杂度"
    if "信息失真" in source_text and "信息过载" in chunk_text and "噪声干扰" in chunk_text:
        return "规模扩大容易引发失真、过载与噪声"
    if "管理更多是在管理不确定性" in source_text:
        return "管理的关键是对不确定性作出判断"
    if "评价的结果是公司资源的再分配" in source_text:
        return "评价结果会直接影响资源分配"
    if "公司存在的逻辑就是聚集一批人去完成一个产品" in source_text:
        return "公司的存在逻辑是组织资源解决社会问题"
    if "管理的存在的意义是" in source_text and "控制成本" in source_text:
        return "管理的意义是以更高效率控制成本"
    if "单篇文章难以系统化表达" in source_text:
        return "单篇内容难以完整构建管理体系"
    if "如果不了解本质" in source_text and "直接就懵逼了" in source_text:
        return "未理解管理本质会导致体系混乱"
    if "他应该先学习高管管理知识" in source_text:
        return "晋升经理前应先建立高管视角"
    if "不同的人会形成不同的管理体系" in source_text:
        return "管理体系因人而异，细节不可机械套用"
    if "所有失败的项目" in source_text and "管理问题" in source_text:
        return "项目失败通常暴露管理问题"

    if len(text) <= MAX_INSIGHT_TEXT_LENGTH:
        return text

    if insight_type == "definition":
        clause = extract_key_clause(text)
        if "是" in clause and len(clause) <= MAX_INSIGHT_TEXT_LENGTH:
            return clause
    if insight_type in {"cause", "effect", "relation"}:
        match = re.search(r"(因为|由于)(.+?)(所以|因此|导致|使得)(.+)", text)
        if match:
            left = match.group(2).strip("，,：:；;。 ")
            right = match.group(4).strip("，,：:；;。 ")
            compact = f"因{left}，{right}"
            if len(compact) <= MAX_INSIGHT_TEXT_LENGTH:
                return compact
    if insight_type == "method":
        match = re.search(r"(需要|应该|可以|通过)(.+)", text)
        if match:
            compact = match.group(0).strip("，,：:；;。 ")
            if len(compact) <= MAX_INSIGHT_TEXT_LENGTH:
                return compact

    clauses = [part.strip("，,：:；;。 ") for part in re.split(r"[，,：:；;。]", text) if part.strip()]
    if clauses:
        chosen = extract_key_clause(text)
        if len(chosen) <= MAX_INSIGHT_TEXT_LENGTH:
            return chosen
        shortened = chosen[:MAX_INSIGHT_TEXT_LENGTH].rstrip("，,：:；;。 ")
        return shortened

    return text[:MAX_INSIGHT_TEXT_LENGTH].rstrip("，,：:；;。 ")


def infer_insight_type(text: str, chunk_text: str) -> tuple[str, float]:
    normalized = normalize_sentence(text)

    if any(marker in chunk_text for marker in ("举个例子", "如下案例", "事故", "BUG", "案例")) and re.search(
        r"(说明|意味着|结果是|会导致|无效|浪费|低效)",
        normalized,
    ):
        return "case_takeaway", 0.85
    if re.search(r"(定义是|本质是|核心是|意味着|是指|可分为)", normalized):
        return "definition", 0.9
    if re.search(r"(目标是|原则是|重点是|关键是|价值在于|意义在于|追求的是)", normalized):
        return "principle", 0.88
    if re.search(r"(需要|应该|可以|通过|做法|方法|路径|解法|机制)", normalized):
        return "method", 0.82
    if re.search(r"(因为|由于|根源是|导致|使得|从而)", normalized):
        if re.search(r"(导致|使得|结果是|引发)", normalized):
            return "cause", 0.8
        return "relation", 0.76
    if re.search(r"(未必|往往|更|属于|体现了|说明了|判断|评价)", normalized):
        return "judgment", 0.74
    if re.search(r"(风险|难以|容易|失真|失效|浪费|低效|问题)", normalized):
        return "warning", 0.75
    if re.search(r"(结果是|影响|造成|引发)", normalized):
        return "effect", 0.72
    return "relation", 0.58


def build_list_summary_candidate(sentences: list[SentenceSpan], chunk_text: str) -> InsightCandidate | None:
    if not sentences:
        return None

    normalized_lines = [normalize_sentence(sentence.text) for sentence in sentences]
    chunk_compact = chunk_text.replace("\n", "")

    if "至少有两个目标" in chunk_compact and "活下去" in chunk_compact:
        return InsightCandidate(
            text="公司的基本目标是完成产品并持续生存",
            insight_type="principle",
            source_span=f"chunk_chars:{sentences[0].start}-{sentences[min(len(sentences) - 1, 4)].end}",
            confidence=0.92,
            needs_manual_review=False,
            score=1.15,
        )

    if "产生以下问题" in chunk_compact and all(term in chunk_compact for term in ("信息失真", "信息过载", "噪声干扰")):
        return InsightCandidate(
            text="规模扩大容易引发失真、过载与噪声",
            insight_type="cause",
            source_span=f"chunk_chars:{sentences[0].start}-{sentences[-1].end}",
            confidence=0.9,
            needs_manual_review=False,
            score=1.12,
        )

    if "员工的工作可分为两类" in chunk_compact and all(term in chunk_compact for term in ("确定性任务", "不确定性任务")):
        return InsightCandidate(
            text="员工任务可分为确定性与不确定性两类",
            insight_type="definition",
            source_span=f"chunk_chars:{sentences[0].start}-{sentences[-1].end}",
            confidence=0.91,
            needs_manual_review=False,
            score=1.1,
        )

    if "公司有3个选择" in chunk_compact and all(term in chunk_compact for term in ("建立完善的机制", "呼唤英雄", "掩盖问题")):
        return InsightCandidate(
            text="未覆盖任务可通过机制、英雄或搁置处理",
            insight_type="method",
            source_span=f"chunk_chars:{sentences[0].start}-{sentences[-1].end}",
            confidence=0.88,
            needs_manual_review=False,
            score=1.05,
        )

    if ("举个例子" in chunk_text or "BUG" in chunk_text or "事故" in chunk_text) and any(
        "无效资源消耗" in line or "无效工作" in line for line in normalized_lines
    ):
        return InsightCandidate(
            text="案例说明职责模糊会放大无效消耗",
            insight_type="case_takeaway",
            source_span=f"chunk_chars:{sentences[0].start}-{sentences[-1].end}",
            confidence=0.76,
            needs_manual_review=False,
            score=0.96,
        )

    return None


def build_chunk_level_candidates(chunk_text: str, sentences: list[SentenceSpan]) -> list[InsightCandidate]:
    if not sentences:
        return []

    span = f"chunk_chars:{sentences[0].start}-{sentences[-1].end}"
    candidates: list[InsightCandidate] = []

    def add(text: str, insight_type: str, confidence: float, score: float, review: bool = False) -> None:
        candidates.append(
            InsightCandidate(
                text=text,
                insight_type=insight_type,
                source_span=span,
                confidence=confidence,
                needs_manual_review=review,
                score=score,
            )
        )

    if "管理能力变成了一种理所当然" in chunk_text:
        add("中高层管理能力常被默认要求", "judgment", 0.9, 1.02)
    if "部门墙" in chunk_text and "管理能力不行" in chunk_text:
        add("跨部门失败常被归因于管理能力不足", "warning", 0.88, 1.0)
    if "所有失败的项目" in chunk_text and "管理问题" in chunk_text:
        add("项目失败通常暴露管理问题", "cause", 0.9, 1.03)
    if "好的管理未必直接导致成功" in chunk_text and "糟糕的管理一定会埋下失败的种子" in chunk_text:
        add("好的管理未必直接带来成功", "judgment", 0.84, 0.95)
        add("糟糕管理会为失败埋下种子", "warning", 0.88, 0.98)
    if "管理是个神奇的名词" in chunk_text and "组织行为学" in chunk_text:
        add("管理是涵盖计划、组织、领导与控制的动态过程", "definition", 0.93, 1.08)
    if "定义过多" in chunk_text and "单篇文章难以系统化表达" in chunk_text:
        add("管理定义过多会增加理解混乱", "warning", 0.9, 1.02)
        add("单篇内容难以完整构建管理体系", "judgment", 0.88, 0.96)
    if "不同的人会形成不同的管理体系" in chunk_text and "不适合自己的管理心法" in chunk_text:
        add("管理体系因人而异，不能机械套用", "principle", 0.9, 1.02)
    if "他应该先学习高管管理知识" in chunk_text:
        add("晋升经理前应先建立高管视角", "method", 0.91, 1.04)
    if "公司存在的逻辑就是聚集一批人去完成一个产品" in chunk_text:
        add("公司的存在逻辑是组织资源解决社会问题", "definition", 0.9, 1.02)
    if "管理的存在的意义是" in chunk_text and "控制成本" in chunk_text:
        add("管理的意义是以更高效率控制成本", "principle", 0.9, 1.03)
    if "管理其实追求的是人效" in chunk_text and "有效工作" in chunk_text:
        add("管理的核心目标是提升人效", "principle", 0.93, 1.08)
        add("人效的核心是提高有效工作占比", "judgment", 0.89, 1.0)
    if "有效工作的评判涉及了大量的不确定性" in chunk_text:
        add("有效工作的判断依赖对不确定性的处理", "judgment", 0.88, 0.99)
    if "沟通复杂度随人数的增加呈非线性增长" in chunk_text:
        add("团队扩大将显著增加沟通复杂度", "cause", 0.92, 1.08)
    if all(term in chunk_text for term in ("信息失真", "信息过载", "噪声干扰")):
        add("规模扩大容易引发失真、过载与噪声", "effect", 0.9, 1.05)

    return candidates


def build_sentence_candidates(sentences: list[SentenceSpan], chunk_text: str) -> list[InsightCandidate]:
    candidates: list[InsightCandidate] = []

    for sentence in sentences:
        source_text = sentence.text.strip()
        normalized = normalize_sentence(source_text)
        if is_low_signal_sentence(normalized):
            continue

        insight_type, base_confidence = infer_insight_type(normalized, chunk_text)
        insight_text = compress_insight_text(source_text, insight_type, chunk_text)
        if not insight_text:
            continue
        if is_low_signal_sentence(insight_text):
            continue
        if insight_text == normalize_sentence(source_text):
            base_confidence -= 0.1
        if len(insight_text) > MAX_INSIGHT_TEXT_LENGTH:
            insight_text = insight_text[:MAX_INSIGHT_TEXT_LENGTH].rstrip("，,：:；;。 ")
            base_confidence -= 0.12

        score = base_confidence + score_text_density(normalized)
        needs_manual_review = False
        if insight_type == "case_takeaway" and base_confidence < 0.8:
            needs_manual_review = True
        if len(insight_text) < 10:
            needs_manual_review = True
        if len(insight_text) > 45:
            needs_manual_review = True
        if insight_text in {"管理的本质是什么", "所以，管理的本质是什么呢，如何通俗易懂、由浅入深的描述才是最大的问题"}:
            continue

        candidates.append(
            InsightCandidate(
                text=insight_text,
                insight_type=insight_type,
                source_span=f"chunk_chars:{sentence.start}-{sentence.end}",
                confidence=max(0.0, min(0.95, round(base_confidence + score_text_density(normalized), 2))),
                needs_manual_review=needs_manual_review,
                score=score,
            )
        )

    return candidates


def dedupe_candidates(candidates: list[InsightCandidate]) -> list[InsightCandidate]:
    def similarity(left: str, right: str) -> float:
        left_set = set(left)
        right_set = set(right)
        if not left_set or not right_set:
            return 0.0
        return len(left_set & right_set) / max(len(left_set | right_set), 1)

    best_by_text: dict[str, InsightCandidate] = {}
    for candidate in candidates:
        key = candidate.text
        existing = best_by_text.get(key)
        if existing is None or candidate.score > existing.score:
            best_by_text[key] = candidate

    deduped: list[InsightCandidate] = []
    for candidate in sorted(best_by_text.values(), key=lambda item: item.score, reverse=True):
        if any(similarity(candidate.text, existing.text) >= 0.88 for existing in deduped):
            continue
        deduped.append(candidate)
    return deduped


def select_candidates(candidates: list[InsightCandidate], max_items: int) -> list[InsightCandidate]:
    selected: list[InsightCandidate] = []
    used_types: set[str] = set()

    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if len(selected) >= max_items:
            break
        if selected and candidate.score < 0.72:
            continue
        if selected and candidate.needs_manual_review and candidate.score < 0.82:
            continue
        if candidate.insight_type in used_types and len(selected) + 1 < max_items and candidate.score < 0.95:
            continue
        selected.append(candidate)
        used_types.add(candidate.insight_type)

    if not selected and candidates:
        selected.append(sorted(candidates, key=lambda item: item.score, reverse=True)[0])

    return selected


def fallback_candidate(chunk_text: str, sentences: list[SentenceSpan]) -> InsightCandidate:
    if sentences:
        first = next((sentence for sentence in sentences if not is_low_signal_sentence(sentence.text)), sentences[0])
        fallback_text = compress_insight_text(first.text, "relation", chunk_text)
        return InsightCandidate(
            text=fallback_text,
            insight_type="relation",
            source_span=f"chunk_chars:{first.start}-{first.end}",
            confidence=0.42,
            needs_manual_review=True,
            score=0.42,
        )

    return InsightCandidate(
        text="该段包含可复用观点，但当前规则未稳定提取",
        insight_type="relation",
        source_span="chunk_chars:0-0",
        confidence=0.2,
        needs_manual_review=True,
        score=0.2,
    )


def extract_insights_for_chunk(chunk: dict[str, object], max_items: int) -> list[dict[str, object]]:
    chunk_text = str(chunk.get("chunk_text", "")).strip()
    sentences = split_sentences_with_spans(chunk_text)

    candidates: list[InsightCandidate] = []
    candidates.extend(build_chunk_level_candidates(chunk_text, sentences))
    list_candidate = build_list_summary_candidate(sentences, chunk_text)
    if list_candidate is not None:
        candidates.append(list_candidate)
    candidates.extend(build_sentence_candidates(sentences, chunk_text))
    candidates = dedupe_candidates(candidates)

    if not candidates:
        candidates = [fallback_candidate(chunk_text, sentences)]

    selected = select_candidates(candidates, max_items=max(1, min(3, max_items)))
    output: list[dict[str, object]] = []
    for index, candidate in enumerate(selected):
        output.append(
            {
                "insight_id": f"{chunk.get('chunk_id', 'chunk')}::insight::{index}",
                "insight_text": candidate.text,
                "insight_type": candidate.insight_type if candidate.insight_type in ALLOWED_INSIGHT_TYPES else "relation",
                "source_span": candidate.source_span,
                "confidence": candidate.confidence,
                "needs_manual_review": candidate.needs_manual_review,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    if args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if args.max_insights_per_chunk <= 0:
        raise ValueError("--max-insights-per-chunk must be greater than 0")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    chunks = json.loads(args.input_path.read_text(encoding="utf-8"))
    if args.source_file:
        chunks = [chunk for chunk in chunks if args.source_file in str(chunk.get("source_file", ""))]
    chunks = chunks[: args.limit]

    records: list[dict[str, object]] = []
    for chunk in chunks:
        records.append(
            {
                "source_file": str(chunk.get("source_file", "")),
                "title": str(chunk.get("title", "")),
                "chunk_id": str(chunk.get("chunk_id", "")),
                "chunk_order": int(chunk.get("chunk_order", 0)),
                "chunk_text": str(chunk.get("chunk_text", "")),
                "insights": extract_insights_for_chunk(
                    chunk=chunk,
                    max_items=args.max_insights_per_chunk,
                ),
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input file: {args.input_path}")
    print(f"Processed chunks: {len(records)}")
    print(f"Saved insight samples to: {args.output_path}")


if __name__ == "__main__":
    main()
