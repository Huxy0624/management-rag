#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_PATH = Path("data/pipeline_candidates/v1/tagged_chunks/records.json")
DEFAULT_OUTPUT_PATH = Path("data/pipeline_candidates/v5/tagged_chunks/records.json")

ANSWER_ROLE_VOCAB = (
    "definition",
    "cause",
    "mechanism",
    "solution",
    "principle",
    "example",
    "warning",
    "comparison",
    "summary",
    "none",
)

ROOT_ISSUE_VOCAB = (
    "information_distortion",
    "evaluation_failure",
    "mixed",
    "none",
)

GOV_MODE_VOCAB = (
    "rule_of_man",
    "rule_of_law",
    "hybrid",
    "none",
)

INTENT_VOCAB = (
    "selection",
    "succession",
    "resource_coordination",
    "coaching",
    "mechanism_design",
    "none",
)

SOLUTION_TYPE_VOCAB = (
    "energy_extension",
    "priority_alignment",
    "risk_escalation",
    "mechanism_building",
    "authority_borrowing",
    "responsibility_binding",
    "none",
)

TARGET_ROLE_VOCAB = ("executive", "director", "manager", "employee")
ROLE_PROFILE_VOCAB = (
    "challenger",
    "loyalist",
    "slippery_worker",
    "high_potential",
    "low_capability",
)

REFINED_ANSWER_ROLE_PROMPT = """请判断该 chunk 的主要句子功能，而不是看关键词表面命中。

只在以下 5 个主类型中优先判断：
1. definition: 主要回答“是什么”
2. principle: 主要解释本质/规律/通用原则
3. mechanism: 主要解释为什么会这样运转，存在因果链
4. solution: 主要给出动作方案或做法
5. summary: 主要做总结归纳

判断规则：
- 如果核心是在定义概念，即使带少量解释或例子，仍然标 definition
- 如果核心是在说明本质/规律，而不是给定义，标 principle
- 如果核心是在说明“为什么会这样”，标 mechanism
- 只有明确给出“应该怎么做”，才标 solution
- 仅仅出现“通过/可以/建立”不等于 solution；如果核心还是解释原因，仍标 mechanism
- 纯总结且没有新判断，标 summary
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine answer_role only for governance matrix v4 schema.")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH, help="Input tagged chunk JSON path.")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output tagged JSON path.")
    parser.add_argument("--limit", type=int, help="Optional source file limit for debug runs.")
    parser.add_argument("--source-file", type=str, help="Optional source_file substring filter.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    normalized = text.replace("“", "").replace("”", "")
    normalized = normalized.replace("‘", "").replace("’", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；])", normalize_text(text))
    return [part.strip() for part in parts if part.strip()]


def filter_records(records: list[dict[str, Any]], source_file: str | None, limit: int | None) -> list[dict[str, Any]]:
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


def joined_text(record: dict[str, Any]) -> tuple[str, str, str]:
    title = normalize_text(str(record.get("title", "")))
    chunk_text = normalize_text(str(record.get("chunk_text", "")))
    insights_text = normalize_text(
        " ".join(
            str(item.get("insight_text", ""))
            for item in record.get("insights", [])
            if isinstance(item, dict)
        )
    )
    return title, chunk_text, insights_text


def legacy_text(record: dict[str, Any]) -> str:
    tags = dict(record.get("tags", {}))
    concepts = dict(tags.get("concepts", {}))
    return normalize_text(
        " ".join(
            [
                " ".join(str(item) for item in tags.get("topic_tags", [])),
                " ".join(str(item) for item in tags.get("scenario_tags", [])),
                " ".join(str(item) for item in concepts.get("explicit", [])),
                " ".join(str(item) for item in concepts.get("inferred", [])),
                " ".join(str(item) for item in concepts.get("original_terms", [])),
                str(tags.get("content_type", "")),
            ]
        )
    )


def keyword_hits(text: str, keywords: list[str] | tuple[str, ...]) -> list[str]:
    normalized = compact_text(text)
    return [keyword for keyword in keywords if keyword and keyword in normalized]


def starts_with_summary_marker(text: str) -> bool:
    normalized = normalize_text(text)
    return normalized.startswith(("综上", "总之", "总结", "由此可见", "归根结底"))


def sentence_role_scores(sentence: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    compact = compact_text(sentence)
    scores = {role: 0.0 for role in ANSWER_ROLE_VOCAB}
    hits: dict[str, list[str]] = {role: [] for role in ANSWER_ROLE_VOCAB}

    definition_patterns = ["是指", "指的是", "所谓", "可以理解为", "本质上是", "定义为"]
    principle_patterns = ["本质", "核心在于", "关键在于", "核心是", "规律", "原则", "意义在于"]
    mechanism_patterns = ["因为", "所以", "导致", "引发", "造成", "如果", "那么", "于是", "之所以", "根源"]
    solution_patterns = ["应该", "需要", "要做", "先", "再", "建立", "推进", "处理", "采取", "做法", "步骤", "通过"]
    warning_patterns = ["不要", "不能", "风险", "反噬", "代价", "陷阱", "误区"]
    summary_patterns = ["综上", "总之", "总结", "由此可见"]
    comparison_patterns = ["不是", "而是", "对比", "并非对立", "相反", "平衡"]
    example_patterns = ["案例", "举例", "比如", "例如", "真实案例", "事故"]

    for keyword in keyword_hits(compact, definition_patterns):
        scores["definition"] += 2.8
        hits["definition"].append(keyword)
    if re.search(r".{0,10}(是|叫做|属于).{1,20}", sentence) and not keyword_hits(compact, mechanism_patterns):
        scores["definition"] += 1.6
        hits["definition"].append("copula_pattern")

    for keyword in keyword_hits(compact, principle_patterns):
        scores["principle"] += 2.4
        hits["principle"].append(keyword)

    for keyword in keyword_hits(compact, mechanism_patterns):
        scores["mechanism"] += 2.3
        hits["mechanism"].append(keyword)
        scores["cause"] += 1.7
        hits["cause"].append(keyword)

    for keyword in keyword_hits(compact, solution_patterns):
        scores["solution"] += 1.4
        hits["solution"].append(keyword)

    for keyword in keyword_hits(compact, warning_patterns):
        scores["warning"] += 1.8
        hits["warning"].append(keyword)

    for keyword in keyword_hits(compact, summary_patterns):
        scores["summary"] += 2.0
        hits["summary"].append(keyword)

    for keyword in keyword_hits(compact, comparison_patterns):
        scores["comparison"] += 1.8
        hits["comparison"].append(keyword)

    for keyword in keyword_hits(compact, example_patterns):
        scores["example"] += 1.6
        hits["example"].append(keyword)

    # Function-level constraints.
    if scores["definition"] > 0:
        scores["mechanism"] -= 0.9
        scores["solution"] -= 1.1
    if scores["principle"] > 0 and scores["solution"] > 0:
        scores["solution"] -= 0.8
    if scores["mechanism"] > 0 and scores["solution"] > 0:
        if scores["mechanism"] >= scores["solution"]:
            scores["solution"] -= 1.2
            hits["mechanism"].append("mechanism_over_solution")
        else:
            scores["mechanism"] -= 0.4
    if scores["summary"] > 0 and any(scores[item] >= 2.2 for item in ("definition", "principle", "mechanism", "solution")):
        scores["summary"] -= 1.0
        hits["summary"].append("summary_deprioritized")

    return scores, hits


def choose_answer_role(record: dict[str, Any], title: str, chunk_text: str, insights_text: str) -> tuple[str, dict[str, Any]]:
    scores = {role: 0.0 for role in ANSWER_ROLE_VOCAB}
    hits: dict[str, list[str]] = {role: [] for role in ANSWER_ROLE_VOCAB}

    candidate_sentences: list[tuple[str, str, float]] = []
    for insight in record.get("insights", []):
        if isinstance(insight, dict):
            candidate_sentences.append((str(insight.get("insight_text", "")), "insight", 1.3))
    for sentence in split_sentences(chunk_text)[:6]:
        candidate_sentences.append((sentence, "chunk", 1.0))
    if title:
        candidate_sentences.append((title, "title", 0.8))

    strongest_sentence = {"text": "", "source": "", "role": "none", "score": 0.0}

    for sentence, source, weight in candidate_sentences:
        sentence_scores, sentence_hits = sentence_role_scores(sentence)
        top_role, top_score = max(sentence_scores.items(), key=lambda item: (item[1], -ANSWER_ROLE_VOCAB.index(item[0])))
        if top_score * weight > strongest_sentence["score"]:
            strongest_sentence = {
                "text": sentence,
                "source": source,
                "role": top_role,
                "score": round(top_score * weight, 4),
            }
        for role, score in sentence_scores.items():
            scores[role] += score * weight
            if sentence_hits[role]:
                hits[role].extend(f"{source}:{item}" for item in sentence_hits[role][:3])

    full_text = compact_text(" ".join([title, insights_text, chunk_text]))
    if "什么是" in full_text:
        scores["definition"] += 1.2
        scores["mechanism"] -= 0.9
        scores["solution"] -= 1.0
        hits["definition"].append("global:what_is_priority")
    if "本质" in full_text and scores["definition"] < 3.0:
        scores["principle"] += 1.2
        hits["principle"].append("global:essence_priority")
    if any(token in full_text for token in ("为什么", "为何")):
        scores["mechanism"] += 0.8
        scores["cause"] += 0.6
        scores["solution"] -= 0.9
        scores["definition"] -= 0.8
        hits["mechanism"].append("global:why_priority")
    if any(token in full_text for token in ("怎么", "如何", "应该")):
        if scores["mechanism"] > 0:
            scores["solution"] += 0.4
            hits["solution"].append("global:how_secondary")
        else:
            scores["solution"] += 1.0
            hits["solution"].append("global:how_primary")
        scores["definition"] -= 0.5
    if starts_with_summary_marker(chunk_text):
        scores["summary"] += 1.2
        hits["summary"].append("global:summary_prefix")

    top_role, top_score = max(scores.items(), key=lambda item: (item[1], -ANSWER_ROLE_VOCAB.index(item[0])))
    if top_score < 2.4:
        top_role = "none"
    return top_role, {"scores": scores, "hits": hits, "strongest_sentence": strongest_sentence, "prompt": REFINED_ANSWER_ROLE_PROMPT}


def choose_root_issue(record: dict[str, Any], title: str, chunk_text: str, insights_text: str) -> tuple[str, dict[str, Any]]:
    full_text = compact_text(" ".join([title, insights_text, chunk_text]))
    legacy = compact_text(legacy_text(record))
    info_score = 0.0
    eval_score = 0.0
    info_hits: list[str] = []
    eval_hits: list[str] = []

    for keyword, weight in (
        ("信息失真", 2.2),
        ("信息差", 1.6),
        ("信息传递", 1.6),
        ("沟通复杂度", 1.8),
        ("层级", 1.2),
        ("传导", 1.2),
        ("向上汇报", 1.4),
        ("向下解释", 1.4),
        ("压缩", 1.0),
        ("扩散", 1.0),
    ):
        if keyword in full_text:
            info_score += weight
            info_hits.append(keyword)

    for keyword, weight in (
        ("评价失效", 2.4),
        ("评价标准", 1.8),
        ("评价权", 1.6),
        ("奖惩", 1.5),
        ("激励", 1.3),
        ("资源分配", 1.6),
        ("资源配置", 1.6),
        ("绩效", 1.2),
        ("公平", 1.0),
        ("价值", 0.9),
        ("人效", 0.9),
    ):
        if keyword in full_text:
            eval_score += weight
            eval_hits.append(keyword)

    if "信息失真" in legacy:
        info_score += 1.0
        info_hits.append("legacy:信息失真")
    if "评价失效" in legacy:
        eval_score += 1.0
        eval_hits.append("legacy:评价失效")

    if info_score >= 2.5 and eval_score >= 2.5 and len(info_hits) >= 2 and len(eval_hits) >= 2:
        return "mixed", {"info_score": round(info_score, 2), "eval_score": round(eval_score, 2), "info_hits": info_hits, "eval_hits": eval_hits}
    if info_score >= 2.8 and info_score >= eval_score + 1.2:
        return "information_distortion", {"info_score": round(info_score, 2), "eval_score": round(eval_score, 2), "info_hits": info_hits, "eval_hits": eval_hits}
    if eval_score >= 2.8 and eval_score >= info_score + 1.2:
        return "evaluation_failure", {"info_score": round(info_score, 2), "eval_score": round(eval_score, 2), "info_hits": info_hits, "eval_hits": eval_hits}
    return "none", {"info_score": round(info_score, 2), "eval_score": round(eval_score, 2), "info_hits": info_hits, "eval_hits": eval_hits}


def choose_gov_mode(title: str, chunk_text: str, insights_text: str) -> tuple[str, list[str]]:
    full_text = compact_text(" ".join([title, insights_text, chunk_text]))
    man_hits = keyword_hits(full_text, ["人治", "站台", "拍板", "借力", "投诉", "老板出面", "硬推", "个人魅力", "救火", "临时"])
    law_hits = keyword_hits(full_text, ["法治", "机制", "制度", "流程", "规则", "体系", "复盘", "长期", "标准", "SOP"])
    if man_hits and law_hits:
        return "hybrid", man_hits[:2] + law_hits[:2]
    if law_hits:
        return "rule_of_law", law_hits[:3]
    if man_hits:
        return "rule_of_man", man_hits[:3]
    return "none", []


def choose_intent(answer_role: str, gov_mode: str, title: str, chunk_text: str, insights_text: str) -> tuple[str, dict[str, Any]]:
    full_text = compact_text(" ".join([title, insights_text, chunk_text]))
    score_table: dict[str, float] = {label: 0.0 for label in INTENT_VOCAB}
    hit_table: dict[str, list[str]] = {label: [] for label in INTENT_VOCAB}
    intent_rules = {
        "selection": [("选拔", 1.8), ("候选人", 1.6), ("任命", 1.5), ("提拔", 1.5), ("选人", 1.5)],
        "succession": [("接班", 2.0), ("梯队", 1.8), ("后备", 1.5), ("储备干部", 1.8), ("继任", 1.8)],
        "resource_coordination": [("跨部门", 1.8), ("部门墙", 1.8), ("协作", 1.2), ("协调资源", 1.8), ("依赖方", 1.2), ("排期", 1.0), ("控制权", 1.2)],
        "coaching": [("培养", 1.8), ("辅导", 1.8), ("带人", 1.6), ("反馈", 1.0), ("成长", 1.0), ("训练", 1.2)],
        "mechanism_design": [("机制", 1.4), ("制度", 1.5), ("流程", 1.3), ("规则", 1.3), ("SOP", 1.8), ("复盘", 1.1), ("标准", 1.0), ("体系", 1.0)],
    }
    for intent, items in intent_rules.items():
        for keyword, weight in items:
            if keyword in full_text:
                score_table[intent] += weight
                hit_table[intent].append(keyword)
    if gov_mode == "rule_of_law":
        score_table["mechanism_design"] += 0.8
        hit_table["mechanism_design"].append("derived:rule_of_law")
    if answer_role == "solution" and "跨部门" in full_text:
        score_table["resource_coordination"] += 0.7
        hit_table["resource_coordination"].append("derived:cross_function_solution")
    if answer_role in {"principle", "definition"} and score_table["mechanism_design"] < 2.0:
        score_table["mechanism_design"] -= 0.4
    best_intent, best_score = max(score_table.items(), key=lambda item: (item[1], -INTENT_VOCAB.index(item[0])))
    if best_intent == "none" or best_score < 2.0:
        return "none", {"scores": score_table, "hits": hit_table}
    return best_intent, {"scores": score_table, "hits": hit_table}


def choose_solution_type(answer_role: str, gov_mode: str, intent: str, title: str, chunk_text: str, insights_text: str) -> tuple[str, list[str]]:
    if answer_role not in {"solution", "mechanism", "warning", "example"}:
        return "none", []
    full_text = compact_text(" ".join([title, insights_text, chunk_text]))
    mapping = [
        ("mechanism_building", ["机制", "制度", "规则", "流程", "复盘", "体系", "标准"]),
        ("authority_borrowing", ["领导站台", "老板出面", "借力", "借势", "抄送CEO", "拉着老板"]),
        ("risk_escalation", ["升级", "上升", "暴露风险", "提前汇报", "风险上报", "预警"]),
        ("responsibility_binding", ["责任", "负责", "绑定", "问责", "拍板权", "说了算"]),
        ("priority_alignment", ["优先级", "目标对齐", "聚焦", "最重要", "减少无效工作"]),
        ("energy_extension", ["加班", "冲刺", "扛", "顶住", "多投入"]),
    ]
    for label, keywords in mapping:
        hits = keyword_hits(full_text, keywords)
        if hits:
            return label, hits[:3]
    if gov_mode == "rule_of_law" and intent == "mechanism_design":
        return "mechanism_building", ["derived:mechanism_design"]
    return "none", []


def choose_target_role(title: str, chunk_text: str, insights_text: str) -> tuple[list[str], list[str]]:
    full_text = compact_text(" ".join([title, insights_text, chunk_text]))
    mapping = [
        ("executive", ["高管", "CEO", "VP", "老板"]),
        ("director", ["总监", "中层", "部门负责人"]),
        ("manager", ["经理", "Leader", "项目负责人", "管理者"]),
        ("employee", ["员工", "一线", "骨干", "同学们"]),
    ]
    selected: list[str] = []
    hits: list[str] = []
    for role, keywords in mapping:
        matched = keyword_hits(full_text, keywords)
        if matched:
            selected.append(role)
            hits.extend(f"{role}:{item}" for item in matched[:2])
    return selected[:2], hits[:4]


def choose_role_profile(title: str, chunk_text: str, insights_text: str) -> tuple[list[str], list[str]]:
    full_text = compact_text(" ".join([title, insights_text, chunk_text]))
    mapping = [
        ("challenger", ["挑战", "不服", "吊打", "质疑"]),
        ("loyalist", ["忠诚", "信任", "稳定", "跟随"]),
        ("slippery_worker", ["滚刀肉", "出工不出力", "磨洋工", "软性抵抗"]),
        ("high_potential", ["潜力", "骨干", "培养对象"]),
        ("low_capability", ["能力不足", "不会做", "准备不足"]),
    ]
    selected: list[str] = []
    hits: list[str] = []
    for label, keywords in mapping:
        matched = keyword_hits(full_text, keywords)
        if matched:
            selected.append(label)
            hits.extend(f"{label}:{item}" for item in matched[:2])
    return selected[:2], hits[:4]


def sanitize_vocab(value: str, vocab: tuple[str, ...]) -> str:
    return value if value in vocab else "none"


def sanitize_list(values: list[str], vocab: tuple[str, ...]) -> list[str]:
    output: list[str] = []
    for value in values:
        if value in vocab and value not in output:
            output.append(value)
    return output


def extract_with_rules(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    title, chunk_text, insights_text = joined_text(record)
    answer_role, answer_debug = choose_answer_role(record, title, chunk_text, insights_text)
    root_issue, root_debug = choose_root_issue(record, title, chunk_text, insights_text)
    gov_mode, gov_hits = choose_gov_mode(title, chunk_text, insights_text)
    intent, intent_debug = choose_intent(answer_role, gov_mode, title, chunk_text, insights_text)
    solution_type, solution_hits = choose_solution_type(answer_role, gov_mode, intent, title, chunk_text, insights_text)
    target_role, target_hits = choose_target_role(title, chunk_text, insights_text)
    role_profile, profile_hits = choose_role_profile(title, chunk_text, insights_text)

    tags = {
        "answer_role": sanitize_vocab(answer_role, ANSWER_ROLE_VOCAB),
        "root_issue": sanitize_vocab(root_issue, ROOT_ISSUE_VOCAB),
        "gov_mode": sanitize_vocab(gov_mode, GOV_MODE_VOCAB),
        "intent": sanitize_vocab(intent, INTENT_VOCAB),
        "solution_type": sanitize_vocab(solution_type, SOLUTION_TYPE_VOCAB),
        "target_role": sanitize_list(target_role, TARGET_ROLE_VOCAB),
        "role_profile": sanitize_list(role_profile, ROLE_PROFILE_VOCAB),
    }
    debug = {
        "answer_role": answer_debug,
        "root_issue": root_debug,
        "gov_mode_hits": gov_hits,
        "intent": intent_debug,
        "solution_type_hits": solution_hits,
        "target_role_hits": target_hits,
        "role_profile_hits": profile_hits,
    }
    return tags, debug


def compute_manual_review(tags: dict[str, Any], debug: dict[str, Any]) -> bool:
    none_count = sum(1 for key in ("answer_role", "root_issue", "gov_mode", "intent", "solution_type") if tags.get(key) == "none")
    if none_count >= 4:
        return True
    if tags["answer_role"] == "definition" and tags["root_issue"] == "mixed":
        return True
    if tags["intent"] == "none" and tags["solution_type"] != "none":
        return True
    if tags["answer_role"] == "mechanism" and not debug["root_issue"]["info_hits"] and not debug["root_issue"]["eval_hits"]:
        return True
    return False


def transform_record(record: dict[str, Any]) -> dict[str, Any]:
    governance_tags_v4, tagging_debug = extract_with_rules(record)
    return {
        "source_file": str(record.get("source_file", "")),
        "title": str(record.get("title", "")),
        "chunk_id": str(record.get("chunk_id", "")),
        "chunk_order": int(record.get("chunk_order", 0)),
        "chunk_text": str(record.get("chunk_text", "")),
        "insights": record.get("insights", []),
        "legacy_tags": record.get("tags", {}),
        "governance_tags_v4": governance_tags_v4,
        "tagging_debug": tagging_debug,
        "needs_manual_review": compute_manual_review(governance_tags_v4, tagging_debug),
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0")
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    records = json.loads(args.input_path.read_text(encoding="utf-8"))
    filtered = filter_records(records, args.source_file, args.limit)
    transformed = [transform_record(record) for record in filtered]

    root_issue_counts = Counter(item["governance_tags_v4"]["root_issue"] for item in transformed)
    answer_role_counts = Counter(item["governance_tags_v4"]["answer_role"] for item in transformed)
    intent_counts = Counter(item["governance_tags_v4"]["intent"] for item in transformed)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(transformed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Input file: {args.input_path}")
    print(f"Total input records: {len(records)}")
    print(f"Selected records: {len(filtered)}")
    print(f"Unique sources: {len({item['source_file'] for item in transformed})}")
    print(f"Needs manual review: {sum(1 for item in transformed if item['needs_manual_review'])}")
    print(f"Root issue counts: {dict(root_issue_counts)}")
    print(f"Answer role counts: {dict(answer_role_counts)}")
    print(f"Intent counts: {dict(intent_counts)}")
    print(f"Saved refined governance tags to: {args.output_path}")


if __name__ == "__main__":
    main()
