from __future__ import annotations

import re
from typing import Literal

OutputLanguage = Literal["zh", "en"]


def infer_output_language(text: str) -> OutputLanguage:
    """
    Minimal detector: more Latin letters than CJK characters -> English, else Chinese.
    Does not affect retrieval; only steers final answer language.
    """
    if not (text or "").strip():
        return "zh"
    latin = len(re.findall(r"[A-Za-z]", text))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    if latin > cjk:
        return "en"
    return "zh"


# Shared guidance for English answers (surface + rewrite + minimal RAG).
ENGLISH_ANSWER_GUIDANCE = (
    "【Output language: English】The user's question is primarily in English. "
    "Write the entire answer in natural, fluent English.\n"
    "- Understand the full logic and evidence first; do not translate line-by-line from any Chinese in the JSON or passages.\n"
    "- Preserve causal structure, order of reasoning, emphasis, and judgments exactly as implied by the structured input.\n"
    "- Use clear, idiomatic English; avoid Chinglish, mechanical tone, stacked A/B/C/D lists unless steps are truly required, "
    "and one-sentence-only conclusions that hide reasoning.\n"
    "- Prefer plain vocabulary and varied sentence rhythm; do not show off with long or ornate sentences.\n"
    "- You may smooth phrasing for clarity, but do not change meaning or add new claims.\n"
    "- Leave enough reasoning visible so a reader can follow how you reached the conclusion.\n"
)


def answer_language_suffix(output_language: str) -> str:
    """Appended to the model user message after the main JSON / instructions."""
    if output_language == "en":
        return "\n\n" + ENGLISH_ANSWER_GUIDANCE
    return "\n\n请用中文撰写完整答案正文（保持与 planner 逻辑一致，不要逐句硬译英文证据）。\n"


ZH_MINIMAL_TAIL = "请用中文撰写完整答案。"
