from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_TITLE = "TraceAnswer"
DEFAULT_TAGLINE = "An AI Q&A demo that makes retrieval, controlled generation, and fallback behavior easier to inspect."
DEFAULT_DESCRIPTION = (
    "This demo combines retrieval, planner-based generation, controlled answering, and fallback safeguards "
    "into a single shareable experience. It is built to make system behavior easier to inspect, evaluate, and improve."
)
DEFAULT_EXAMPLES = [
    "什么是跨部门协作问题？",
    "为什么跨部门协作经常低效？",
    "怎么临时推进跨部门合作？",
    "怎么通过机制解决跨部门协作？",
    "为什么信息不通会引发跨部门冲突？",
    "老板问进度时，向上汇报应该怎么压缩信息？",
    "战略传到一线时，向下解释应该怎么做？",
    "管理中，临时推进和机制建设有什么区别？",
]


@dataclass(frozen=True)
class DemoConfig:
    title: str
    tagline: str
    description: str
    examples: list[str]
    debug_token: str | None
    admin_token: str | None
    show_debug_toggle: bool
    rate_limit_window_seconds: int
    rate_limit_max_requests: int
    input_placeholder: str
    answer_title: str
    feedback_prompt: str
    debug_title: str
    debug_description: str


def _split_examples(value: str | None) -> list[str]:
    if not value:
        return DEFAULT_EXAMPLES
    parts = [item.strip() for item in value.split("||")]
    return [item for item in parts if item]


def get_demo_config() -> DemoConfig:
    return DemoConfig(
        title=os.getenv("DEMO_TITLE", DEFAULT_TITLE),
        tagline=os.getenv("DEMO_TAGLINE", DEFAULT_TAGLINE),
        description=os.getenv("DEMO_DESCRIPTION", DEFAULT_DESCRIPTION),
        examples=_split_examples(os.getenv("DEMO_EXAMPLES")),
        debug_token=os.getenv("DEMO_DEBUG_TOKEN"),
        admin_token=os.getenv("DEMO_ADMIN_TOKEN"),
        show_debug_toggle=os.getenv("DEMO_SHOW_DEBUG_TOGGLE", "true").lower() in {"1", "true", "yes", "on"},
        rate_limit_window_seconds=int(os.getenv("DEMO_RATE_LIMIT_WINDOW_SECONDS", "60")),
        rate_limit_max_requests=int(os.getenv("DEMO_RATE_LIMIT_MAX_REQUESTS", "12")),
        input_placeholder=os.getenv("DEMO_INPUT_PLACEHOLDER", "例如：怎么通过机制解决跨部门协作？"),
        answer_title=os.getenv("DEMO_ANSWER_TITLE", "Answer"),
        feedback_prompt=os.getenv("DEMO_FEEDBACK_PROMPT", "Was this response helpful for your use case?"),
        debug_title=os.getenv("DEMO_DEBUG_TITLE", "Debug View"),
        debug_description=os.getenv(
            "DEMO_DEBUG_DESCRIPTION",
            "Visible only in debug mode. Shows retrieval, planner output, selector path, and stage timings.",
        ),
    )
