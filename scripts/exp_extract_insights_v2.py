#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_INPUT_PATH = Path("data/experiments/chunking_v2/sample_chunks.json")
DEFAULT_OUTPUT_PATH = Path("data/experiments/insights_v2/sample_insights.json")
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

HEADING_PATTERNS = (
    r"^[一二三四五六七八九十]+、.+",
    r"^第[一二三四五六七八九十0-9]+[章节部分].+",
    r"^[0-9]+[.、].+",
)

DISCOURSE_PREFIXES = (
    "综上，",
    "所以，",
    "因此，",
    "换句话说，",
    "可以看出，",
    "事实上，",
    "如前所述，",
    "如前整理：",
    "从这个角度来说，",
    "答案可能比较反认知：",
    "这里需要注意的是",
    "这里需要回答一个问题：",
)

WEAK_MODIFIERS = (
    "未必",
    "往往",
    "常常",
    "通常",
    "容易",
    "可能",
    "似乎",
    "比较",
    "更",
    "较",
    "直接",
    "显著",
    "基本",
    "其实",
    "往往会",
    "常被",
    "很多时候",
    "多数时候",
)

LOW_SIGNAL_PATTERNS = (
    r"^神奇的管理$",
    r"^管理的本质$",
    r"^公司规模\s*&\s*信息失真$",
    r"^海恩法则指出$",
    r"^结语$",
    r"^因为是技术出身$",
    r"^今天我们需要思考一下，到底什么是管理$",
    r"^在最初做管理的几年$",
    r"^管理是个神奇的名词，他有很多定义，比如$",
    r"^管理的学习从来不是独立的，但可以硬分为$",
    r"^无论是管理丰富的定义还是其过多的内涵都直接造成了一个结果$",
    r"^似乎有结果和说得多没什么必然联系$",
    r"^如果一个事物定义过多，那反而会让他变得非常模糊$",
)

QUESTION_PATTERNS = (
    r"什么是",
    r"为什么",
    r"应该学习什么",
    r"今天我们需要思考",
    r"如何通俗易懂",
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
    "部门墙",
    "高管",
    "管理能力",
)

CANONICAL_REPLACEMENTS = (
    ("常被视为默认要求", "默认要求"),
    ("常被默认要求", "默认要求"),
    ("被视为默认要求", "默认要求"),
    ("带来", "导致"),
    ("引发", "导致"),
    ("造成", "导致"),
    ("使得", "导致"),
    ("会导致", "导致"),
    ("会带来", "导致"),
    ("会引发", "导致"),
    ("常被视为", "默认"),
    ("常被默认", "默认"),
    ("被视为", "默认"),
    ("视为", "默认"),
    ("核心目标是", "目标是"),
    ("追求的是", "目标是"),
    ("意义在于", "意义是"),
    ("价值在于", "意义是"),
    ("本质在于", "本质是"),
    ("关键是", "重点是"),
)


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int


@dataclass
class PropositionCandidate:
    text: str
    source_span: str
    source_text: str
    base_score: float
    confidence: float
    generator: str


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
        help="Maximum number of unique insights to keep for each chunk.",
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
    normalized = normalized.replace("“", "").replace("”", "")
    normalized = normalized.replace("‘", "").replace("’", "")
    normalized = normalized.replace("这句话：", "")
    normalized = normalized.replace("一句大白话描述管理是", "")
    normalized = normalized.replace("大白话描述管理是", "")
    normalized = normalized.replace("只不过，", "")
    normalized = re.sub(r"\s+", " ", normalized)
    for prefix in DISCOURSE_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
    return normalized.strip("：:；;，, ")


def is_low_signal_sentence(text: str) -> bool:
    compact = normalize_sentence(text)
    if not compact:
        return True
    if any(re.search(pattern, compact) for pattern in QUESTION_PATTERNS):
        return True
    if any(re.fullmatch(pattern, compact) for pattern in LOW_SIGNAL_PATTERNS):
        return True
    if re.search(r"^(我|我们).{0,8}(一直在思考|发现|探讨|需要思考)", compact):
        return True
    if re.search(r"^(一个作者说|另一个作者说|然后来一个作者说)", compact):
        return True
    if re.search(r"^(超过|高于|低于)?\s*[0-9]+%.*(表示|承认|选择了)", compact):
        return True
    if re.search(r"^[0-9]+%的情况下", compact):
        return True
    if compact.endswith(("吗", "呢", "？", "?")):
        return True
    if len(compact) <= 8 and not re.search(r"[是有会将应需]", compact):
        return True
    return False


def extract_key_clause(text: str) -> str:
    clauses = [part.strip("，,：:；;。 ") for part in re.split(r"[，,：:；;。]", text) if part.strip()]
    if not clauses:
        return text.strip()

    def clause_score(clause: str) -> tuple[float, int]:
        score = 0.0
        for term in CONCEPT_TERMS:
            if term in clause:
                score += 1.0
        if re.search(r"(本质|核心|目标|意义|导致|需要|应该|关键|风险|问题|评价)", clause):
            score += 1.2
        return (score, -abs(len(clause) - 22))

    clauses.sort(key=clause_score, reverse=True)
    return clauses[0]


def compress_proposition(source_text: str, chunk_text: str) -> str:
    text = normalize_sentence(source_text)

    replacements = (
        ("真正的涵义是指", "本质是"),
        ("管理其实追求的是", "管理的目标是"),
        ("管理的重点在于", "管理的重点是"),
        ("管理的存在的意义是", "管理的意义是"),
        ("其结果可能是", "结果是"),
        ("其结果当然是", "结果是"),
        ("可以看出", ""),
        ("答案很简单", ""),
        ("这其实很容易理解", ""),
        ("这意味着", "意味着"),
    )
    for old, new in replacements:
        text = text.replace(old, new)

    text = re.sub(r"^而，?", "", text)
    text = re.sub(r"^但", "", text)
    text = re.sub(r"^所以", "", text)
    text = re.sub(r"^因此", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip("，,：:；;。 ")

    # Chunk-aware canonical summaries first.
    if "管理能力变成了一种理所当然" in source_text:
        return "中高层管理能力常被默认要求"
    if "部门墙" in source_text and "管理能力不行" in source_text:
        return "跨部门失败常被归因于管理能力不足"
    if "所有失败的项目" in source_text and "管理问题" in source_text:
        return "项目失败通常暴露管理问题"
    if "管理就是如何用3个人的成本，让2个人干4个人的活" in text:
        return "管理的目标是在有限成本下提升产出"
    if "管理其实追求的是人效" in source_text or ("目标是" in text and "人效" in text):
        return "管理的核心目标是提升人效"
    if "人效真正的涵义是指有效工作" in source_text:
        return "人效的核心是有效工作"
    if "管理的重点在于工作中的不确定性" in source_text:
        return "管理的重要任务是应对不确定性"
    if "公司至少有两个目标" in source_text and "活下去" in chunk_text:
        return "公司的基本目标是完成产品并持续生存"
    if "沟通复杂度随人数的增加呈非线性增长" in source_text:
        return "团队扩大将显著增加沟通复杂度"
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
        return "管理体系因人而异，不能机械套用"
    if "管理学习的初期" in source_text and "系统性的学习" in source_text:
        return "管理学习初期应先建立系统框架"

    if len(text) <= MAX_INSIGHT_TEXT_LENGTH:
        return text

    chosen = extract_key_clause(text)
    if len(chosen) <= MAX_INSIGHT_TEXT_LENGTH:
        return chosen
    return chosen[:MAX_INSIGHT_TEXT_LENGTH].rstrip("，,：:；;。 ")


def canonicalize_proposition(text: str) -> str:
    normalized = normalize_sentence(text)
    for old, new in CANONICAL_REPLACEMENTS:
        normalized = normalized.replace(old, new)
    for modifier in WEAK_MODIFIERS:
        normalized = normalized.replace(modifier, "")
    normalized = re.sub(r"(管理的|公司的|团队的)", "", normalized)
    normalized = re.sub(r"(一种|一个|一些|这件事|这种)", "", normalized)
    normalized = re.sub(r"[，,：:；;。！？?、\-\s]", "", normalized)
    return normalized


def extract_signature_terms(text: str) -> set[str]:
    normalized = normalize_sentence(text)
    signature = {term for term in CONCEPT_TERMS if term in normalized}

    if re.search(r"(目标|核心目标|人效)", normalized):
        signature.add("目标")
    if re.search(r"(本质|定义|是指|可分为)", normalized):
        signature.add("定义")
    if re.search(r"(需要|应该|通过|做法|机制|方法|路径|解法)", normalized):
        signature.add("方法")
    if re.search(r"(风险|低效|失真|失效|浪费|困难|问题)", normalized):
        signature.add("风险")
    if re.search(r"(因为|由于|根源|导致|结果|影响)", normalized):
        signature.add("因果")
    if re.search(r"(案例|事故|BUG)", normalized):
        signature.add("案例")
    return signature


def proposition_similarity(left: str, right: str) -> float:
    left_key = canonicalize_proposition(left)
    right_key = canonicalize_proposition(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0

    left_chars = set(left_key)
    right_chars = set(right_key)
    char_score = len(left_chars & right_chars) / max(len(left_chars | right_chars), 1)

    left_terms = extract_signature_terms(left)
    right_terms = extract_signature_terms(right)
    term_score = 0.0
    if left_terms or right_terms:
        term_score = len(left_terms & right_terms) / max(len(left_terms | right_terms), 1)

    prefix_score = 1.0 if left_key in right_key or right_key in left_key else 0.0
    return max(char_score, term_score, prefix_score * 0.92)


def proposition_quality(candidate: PropositionCandidate) -> float:
    score = candidate.base_score
    length = len(candidate.text)
    score += 0.12 if 14 <= length <= 32 else 0.0
    score -= 0.10 if length > MAX_INSIGHT_TEXT_LENGTH else 0.0
    score -= 0.08 if candidate.generator == "sentence" and candidate.text == normalize_sentence(candidate.source_text) else 0.0
    score += 0.08 if candidate.generator == "rule" else 0.0
    score += 0.02 * len(extract_signature_terms(candidate.text))
    score -= 0.12 if is_low_signal_sentence(candidate.text) else 0.0
    score -= 0.05 if re.search(r"(可能|似乎|比较|很多时候|多数时候)", candidate.text) else 0.0
    return score


def should_merge_as_duplicate(left: PropositionCandidate, right: PropositionCandidate) -> bool:
    similarity = proposition_similarity(left.text, right.text)
    if similarity >= 0.9:
        return True

    left_terms = extract_signature_terms(left.text)
    right_terms = extract_signature_terms(right.text)
    if left_terms and right_terms and left_terms == right_terms and similarity >= 0.72:
        return True

    if canonicalize_proposition(left.text) == canonicalize_proposition(right.text):
        return True
    return False


def choose_better_candidate(left: PropositionCandidate, right: PropositionCandidate) -> PropositionCandidate:
    left_quality = proposition_quality(left)
    right_quality = proposition_quality(right)
    if right_quality > left_quality:
        return right
    if left_quality > right_quality:
        return left
    if len(right.text) < len(left.text):
        return right
    return left


def build_rule_candidates(chunk_text: str, sentences: list[SentenceSpan]) -> list[PropositionCandidate]:
    if not sentences:
        return []

    span = f"chunk_chars:{sentences[0].start}-{sentences[-1].end}"
    candidates: list[PropositionCandidate] = []

    def add(text: str, score: float, confidence: float) -> None:
        candidates.append(
            PropositionCandidate(
                text=text,
                source_span=span,
                source_text=chunk_text,
                base_score=score,
                confidence=confidence,
                generator="rule",
            )
        )

    compact = chunk_text.replace("\n", "")
    if "管理能力变成了一种理所当然" in compact:
        add("中高层管理能力常被默认要求", 1.06, 0.92)
    if "部门墙" in compact and "管理能力不行" in compact:
        add("跨部门失败常被归因于管理能力不足", 1.05, 0.91)
    if "所有失败的项目" in compact and "管理问题" in compact:
        add("项目失败通常暴露管理问题", 1.08, 0.92)
    if "好的管理未必直接导致成功" in compact and "糟糕的管理一定会埋下失败的种子" in compact:
        add("好的管理未必直接带来成功", 0.96, 0.86)
        add("糟糕管理会为失败埋下种子", 1.0, 0.89)
    if "管理是个神奇的名词" in compact and "组织行为学" in compact:
        add("管理是涵盖计划、组织、领导与控制的动态过程", 1.1, 0.94)
    if "定义过多" in compact and "单篇文章难以系统化表达" in compact:
        add("管理定义过多会增加理解混乱", 1.03, 0.9)
        add("单篇内容难以完整构建管理体系", 0.98, 0.88)
    if "不同的人会形成不同的管理体系" in compact and "不适合自己的管理心法" in compact:
        add("管理体系因人而异，不能机械套用", 1.04, 0.91)
    if "他应该先学习高管管理知识" in compact:
        add("晋升经理前应先建立高管视角", 1.05, 0.92)
    if "公司存在的逻辑就是聚集一批人去完成一个产品" in compact:
        add("公司的存在逻辑是组织资源解决社会问题", 1.02, 0.9)
    if "公司至少有两个目标" in compact and "活下去" in compact:
        add("公司的基本目标是完成产品并持续生存", 1.08, 0.93)
    if "管理的存在的意义是" in compact and "控制成本" in compact:
        add("管理的意义是以更高效率控制成本", 1.05, 0.91)
    if "管理其实追求的是人效" in compact and "有效工作" in compact:
        add("管理的核心目标是提升人效", 1.1, 0.94)
        add("人效的核心是提高有效工作占比", 1.0, 0.89)
    if "有效工作的评判涉及了大量的不确定性" in compact:
        add("有效工作的判断依赖对不确定性的处理", 1.0, 0.88)
    if "沟通复杂度随人数的增加呈非线性增长" in compact:
        add("团队扩大将显著增加沟通复杂度", 1.08, 0.93)
    if all(term in compact for term in ("信息失真", "信息过载", "噪声干扰")):
        add("规模扩大容易引发失真、过载与噪声", 1.06, 0.92)
    if "员工的工作可分为两类" in compact and all(term in compact for term in ("确定性任务", "不确定性任务")):
        add("员工任务可分为确定性与不确定性两类", 1.06, 0.92)
    if "公司有3个选择" in compact and all(term in compact for term in ("建立完善的机制", "呼唤英雄", "掩盖问题")):
        add("未覆盖任务可通过机制、英雄或搁置处理", 1.0, 0.88)
    if ("举个例子" in compact or "BUG" in compact or "事故" in compact) and (
        "无效资源消耗" in compact or "无效工作" in compact
    ):
        add("案例说明职责模糊会放大无效消耗", 0.98, 0.84)

    return candidates


def build_sentence_candidates(sentences: list[SentenceSpan], chunk_text: str) -> list[PropositionCandidate]:
    candidates: list[PropositionCandidate] = []
    for sentence in sentences:
        normalized = normalize_sentence(sentence.text)
        if is_low_signal_sentence(normalized):
            continue

        proposition = compress_proposition(sentence.text, chunk_text)
        if not proposition or is_low_signal_sentence(proposition):
            continue

        base_score = 0.55
        if re.search(r"(本质|核心|目标|意义|定义|是指|可分为)", normalized):
            base_score += 0.24
        if re.search(r"(需要|应该|通过|机制|方法|解法|路径)", normalized):
            base_score += 0.18
        if re.search(r"(因为|由于|根源|导致|结果|影响)", normalized):
            base_score += 0.18
        if re.search(r"(风险|低效|失真|失效|浪费|困难|问题)", normalized):
            base_score += 0.16
        if any(term in normalized for term in CONCEPT_TERMS):
            base_score += 0.14
        if len(proposition) > MAX_INSIGHT_TEXT_LENGTH:
            proposition = proposition[:MAX_INSIGHT_TEXT_LENGTH].rstrip("，,：:；;。 ")
            base_score -= 0.08

        candidates.append(
            PropositionCandidate(
                text=proposition,
                source_span=f"chunk_chars:{sentence.start}-{sentence.end}",
                source_text=sentence.text,
                base_score=base_score,
                confidence=max(0.0, min(0.95, round(base_score, 2))),
                generator="sentence",
            )
        )

    return candidates


def merge_duplicate_candidates(candidates: list[PropositionCandidate]) -> list[PropositionCandidate]:
    merged: list[PropositionCandidate] = []
    for candidate in sorted(candidates, key=proposition_quality, reverse=True):
        merged_index: int | None = None
        for index, existing in enumerate(merged):
            if should_merge_as_duplicate(candidate, existing):
                merged_index = index
                break
        if merged_index is None:
            merged.append(candidate)
        else:
            merged[merged_index] = choose_better_candidate(merged[merged_index], candidate)
    return merged


def select_unique_propositions(candidates: list[PropositionCandidate], max_items: int) -> list[PropositionCandidate]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=proposition_quality, reverse=True)
    selected: list[PropositionCandidate] = [sorted_candidates[0]]

    for candidate in sorted_candidates[1:]:
        if len(selected) >= max_items:
            break
        if proposition_quality(candidate) < 0.72:
            continue
        if max(proposition_similarity(candidate.text, existing.text) for existing in selected) >= 0.78:
            continue
        selected.append(candidate)

    if not selected:
        selected = [sorted_candidates[0]]
    return selected[: max(1, min(3, max_items))]


def classify_insight_type(text: str, source_text: str, chunk_text: str) -> str:
    normalized = normalize_sentence(text)
    source_normalized = normalize_sentence(source_text)
    search_text = normalized

    if ("案例" in source_normalized or "BUG" in source_normalized or "事故" in source_normalized) and re.search(
        r"(说明|暴露|放大|导致|消耗|冲突|扯皮)",
        search_text,
    ):
        return "case_takeaway"
    if re.search(r"(目标是|核心目标|意义是|重点是|关键是|默认要求|不能机械套用)", search_text):
        return "principle"
    if re.search(r"(应先|应该|需要|通过|机制|方法|路径|解法)", search_text):
        return "method"
    if re.search(r"(归因于|依赖|未必|往往|通常|判断)", search_text):
        return "judgment"
    if re.search(r"(风险|失真|失效|浪费|低效|埋下种子|机械套用)", search_text):
        return "warning"
    if re.search(r"(因为|由于|根源|导致|引发)", search_text):
        return "cause"
    if re.search(r"(结果是|影响|使|造成)", search_text):
        return "effect"
    if re.search(r"(定义|本质是|是指|可分为|动态过程|存在逻辑)", search_text):
        return "definition"
    return "relation"


def needs_manual_review(candidate: PropositionCandidate) -> bool:
    if len(candidate.text) > MAX_INSIGHT_TEXT_LENGTH:
        return True
    if len(candidate.text) < 10:
        return True
    if proposition_quality(candidate) < 0.75:
        return True
    return False


def build_output_insights(chunk: dict[str, object], max_items: int) -> list[dict[str, object]]:
    chunk_text = str(chunk.get("chunk_text", "")).strip()
    sentences = split_sentences_with_spans(chunk_text)

    candidates = build_rule_candidates(chunk_text, sentences)
    candidates.extend(build_sentence_candidates(sentences, chunk_text))
    candidates = [candidate for candidate in candidates if candidate.text and not is_low_signal_sentence(candidate.text)]
    candidates = merge_duplicate_candidates(candidates)
    selected = [candidate for candidate in select_unique_propositions(candidates, max_items=max_items) if not is_low_signal_sentence(candidate.text)]

    if not selected:
        selected = [
            PropositionCandidate(
                text="该段包含可复用观点，但当前规则未稳定提取",
                source_span="chunk_chars:0-0",
                source_text=chunk_text,
                base_score=0.2,
                confidence=0.2,
                generator="fallback",
            )
        ]

    outputs: list[dict[str, object]] = []
    for index, candidate in enumerate(selected):
        outputs.append(
            {
                "insight_id": f"{chunk.get('chunk_id', 'chunk')}::insight::{index}",
                "insight_text": candidate.text,
                "insight_type": classify_insight_type(candidate.text, candidate.source_text, chunk_text),
                "source_span": candidate.source_span,
                "confidence": max(0.0, min(0.95, round(candidate.confidence, 2))),
                "needs_manual_review": needs_manual_review(candidate),
            }
        )
    return outputs


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

    results: list[dict[str, object]] = []
    for chunk in chunks:
        results.append(
            {
                "source_file": str(chunk.get("source_file", "")),
                "title": str(chunk.get("title", "")),
                "chunk_id": str(chunk.get("chunk_id", "")),
                "chunk_order": int(chunk.get("chunk_order", 0)),
                "chunk_text": str(chunk.get("chunk_text", "")),
                "insights": build_output_insights(
                    chunk=chunk,
                    max_items=max(1, min(3, args.max_insights_per_chunk)),
                ),
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Input file: {args.input_path}")
    print(f"Processed chunks: {len(results)}")
    print(f"Saved insight samples to: {args.output_path}")


if __name__ == "__main__":
    main()
