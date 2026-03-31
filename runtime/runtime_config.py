from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any


PROMPTS_DIR = Path("prompts/generation_v3")
DEFAULT_SURFACE_PROMPT_PATH = PROMPTS_DIR / "answer_prompt_v3.txt"
DEFAULT_REWRITE_PROMPT_PATH = PROMPTS_DIR / "answer_rewrite_prompt_v3.txt"
DEFAULT_SURFACE_MODEL = "gpt-4.1-mini"
DEFAULT_SURFACE_TIMEOUT_SECONDS = 60
DEFAULT_SURFACE_MAX_RETRIES = 2
DEFAULT_SURFACE_RETRY_DELAY_SECONDS = 2
DEFAULT_MAIN_EVIDENCE_COUNT = 2
DEFAULT_SUPPORT_EVIDENCE_COUNT = 2
# Default to a lighter profile suitable for small cloud instances (fewer LLM round-trips).
DEFAULT_RUNTIME_PROFILE = "render_lite"
PROFILE_ENV_VAR = "GENERATION_RUNTIME_PROFILE"

ENV_ENABLE_LLM_SURFACE_GENERATION_V3 = "ENABLE_LLM_SURFACE_GENERATION_V3"
ENV_ENABLE_GENERATION_CHAIN_V2 = "ENABLE_GENERATION_CHAIN_V2"
ENV_ENABLE_CONTROL_CHECKS = "ENABLE_CONTROL_CHECKS"
ENV_ENABLE_REWRITE_V3 = "ENABLE_REWRITE_V3"
ENV_ENABLE_FALLBACK_V21 = "ENABLE_FALLBACK_V21"
ENV_ENABLE_FAILURE_CASE_LOGGER = "ENABLE_FAILURE_CASE_LOGGER"
ENV_FAILURE_CASE_LOG_PATH = "FAILURE_CASE_LOG_PATH"
ENV_DEBUG_RETURN_INTERMEDIATE = "DEBUG_RETURN_INTERMEDIATE"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_BASE_URL = "OPENAI_BASE_URL"
ENV_OPENAI_MODEL = "OPENAI_MODEL"
ENV_OPENAI_TIMEOUT_SECONDS = "OPENAI_TIMEOUT_SECONDS"
ENV_OPENAI_MAX_RETRIES = "OPENAI_MAX_RETRIES"


@dataclass(frozen=True)
class RuntimeProfileDefaults:
    enable_generation_chain_v2: bool
    enable_llm_surface_generation_v3: bool
    enable_control_checks: bool
    enable_rewrite_v3: bool
    enable_fallback_v21: bool
    enable_failure_case_logger: bool
    debug_return_intermediate: bool


RUNTIME_PROFILES: dict[str, RuntimeProfileDefaults] = {
    "local_dev": RuntimeProfileDefaults(
        enable_generation_chain_v2=True,
        enable_llm_surface_generation_v3=True,
        enable_control_checks=True,
        enable_rewrite_v3=True,
        enable_fallback_v21=True,
        enable_failure_case_logger=False,
        debug_return_intermediate=True,
    ),
    "staging": RuntimeProfileDefaults(
        enable_generation_chain_v2=True,
        enable_llm_surface_generation_v3=True,
        enable_control_checks=True,
        enable_rewrite_v3=True,
        enable_fallback_v21=True,
        enable_failure_case_logger=False,
        debug_return_intermediate=False,
    ),
    "production": RuntimeProfileDefaults(
        enable_generation_chain_v2=True,
        enable_llm_surface_generation_v3=True,
        enable_control_checks=True,
        enable_rewrite_v3=True,
        enable_fallback_v21=True,
        enable_failure_case_logger=False,
        debug_return_intermediate=False,
    ),
    "render_lite": RuntimeProfileDefaults(
        enable_generation_chain_v2=True,
        enable_llm_surface_generation_v3=True,
        enable_control_checks=False,
        enable_rewrite_v3=False,
        enable_fallback_v21=False,
        enable_failure_case_logger=False,
        debug_return_intermediate=False,
    ),
    "minimal_rag": RuntimeProfileDefaults(
        enable_generation_chain_v2=False,
        enable_llm_surface_generation_v3=True,
        enable_control_checks=False,
        enable_rewrite_v3=False,
        enable_fallback_v21=False,
        enable_failure_case_logger=False,
        debug_return_intermediate=False,
    ),
}


@dataclass(frozen=True)
class FeatureFlags:
    enable_generation_chain_v2: bool = True
    enable_llm_surface_generation_v3: bool = False
    enable_control_checks: bool = True
    enable_rewrite_v3: bool = True
    enable_fallback_v21: bool = True
    enable_failure_case_logger: bool = False
    debug_return_intermediate: bool = False


@dataclass(frozen=True)
class SurfaceRuntimeConfig:
    api_key: str | None = None
    model_name: str = DEFAULT_SURFACE_MODEL
    prompt_path: Path = DEFAULT_SURFACE_PROMPT_PATH
    rewrite_prompt_path: Path = DEFAULT_REWRITE_PROMPT_PATH
    request_timeout_seconds: int = DEFAULT_SURFACE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_SURFACE_MAX_RETRIES
    retry_delay_seconds: int = DEFAULT_SURFACE_RETRY_DELAY_SECONDS
    base_url: str | None = None


@dataclass(frozen=True)
class PlannerRuntimeConfig:
    main_evidence_count: int = DEFAULT_MAIN_EVIDENCE_COUNT
    support_evidence_count: int = DEFAULT_SUPPORT_EVIDENCE_COUNT


@dataclass(frozen=True)
class RuntimeConfigMetadata:
    profile_name: str
    env_overrides: dict[str, Any]
    cli_overrides: dict[str, Any]


@dataclass(frozen=True)
class GenerationRuntimeConfig:
    metadata: RuntimeConfigMetadata
    feature_flags: FeatureFlags
    surface: SurfaceRuntimeConfig
    planner: PlannerRuntimeConfig
    failure_case_log_path: Path


def parse_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def get_profile_name(args=None) -> str:
    cli_value = getattr(args, "runtime_profile", None) if args is not None else None
    env_value = os.getenv(PROFILE_ENV_VAR)
    profile_name = str(cli_value or env_value or DEFAULT_RUNTIME_PROFILE)
    if profile_name not in RUNTIME_PROFILES:
        supported = ", ".join(sorted(RUNTIME_PROFILES))
        raise ValueError(f"Unsupported runtime profile: {profile_name}. Expected one of: {supported}")
    return profile_name


def _coalesce(cli_value, env_value, default_value):
    if cli_value is not None:
        return cli_value
    if env_value is not None:
        return env_value
    return default_value


def _resolve_bool(args, cli_attr: str, env_name: str, default_value: bool) -> tuple[bool, bool, bool]:
    cli_value = getattr(args, cli_attr, None)
    env_value = parse_bool(os.getenv(env_name))
    value = bool(_coalesce(cli_value, env_value, default_value))
    return value, cli_value is not None, env_value is not None


def _resolve_int(args, cli_attr: str, env_name: str, default_value: int) -> tuple[int, bool, bool]:
    cli_value = getattr(args, cli_attr, None)
    env_value = parse_int(os.getenv(env_name))
    value = int(_coalesce(cli_value, env_value, default_value))
    return value, cli_value is not None, env_value is not None


def _resolve_str(args, cli_attr: str, env_name: str, default_value: str | None) -> tuple[str | None, bool, bool]:
    cli_value = getattr(args, cli_attr, None)
    env_value = os.getenv(env_name)
    value = _coalesce(cli_value, env_value, default_value)
    return None if value is None else str(value), cli_value is not None, env_value is not None


def runtime_config_from_args(args) -> GenerationRuntimeConfig:
    profile_name = get_profile_name(args)
    profile = RUNTIME_PROFILES[profile_name]

    env_overrides: dict[str, Any] = {"profile": profile_name}
    cli_overrides: dict[str, Any] = {}

    enable_generation_chain_v2, cli_set, env_set = _resolve_bool(
        args, "enable_generation_chain_v2", ENV_ENABLE_GENERATION_CHAIN_V2, profile.enable_generation_chain_v2
    )
    if cli_set:
        cli_overrides["enable_generation_chain_v2"] = enable_generation_chain_v2
    elif env_set:
        env_overrides[ENV_ENABLE_GENERATION_CHAIN_V2] = enable_generation_chain_v2

    enable_llm_surface_generation_v3, cli_set, env_set = _resolve_bool(
        args, "enable_llm_surface_generation_v3", ENV_ENABLE_LLM_SURFACE_GENERATION_V3, profile.enable_llm_surface_generation_v3
    )
    if cli_set:
        cli_overrides["enable_llm_surface_generation_v3"] = enable_llm_surface_generation_v3
    elif env_set:
        env_overrides[ENV_ENABLE_LLM_SURFACE_GENERATION_V3] = enable_llm_surface_generation_v3

    enable_control_checks, cli_set, env_set = _resolve_bool(
        args, "enable_control_checks", ENV_ENABLE_CONTROL_CHECKS, profile.enable_control_checks
    )
    if cli_set:
        cli_overrides["enable_control_checks"] = enable_control_checks
    elif env_set:
        env_overrides[ENV_ENABLE_CONTROL_CHECKS] = enable_control_checks

    enable_rewrite_v3, cli_set, env_set = _resolve_bool(
        args, "enable_rewrite_v3", ENV_ENABLE_REWRITE_V3, profile.enable_rewrite_v3
    )
    if cli_set:
        cli_overrides["enable_rewrite_v3"] = enable_rewrite_v3
    elif env_set:
        env_overrides[ENV_ENABLE_REWRITE_V3] = enable_rewrite_v3

    enable_fallback_v21, cli_set, env_set = _resolve_bool(
        args, "enable_fallback_v21", ENV_ENABLE_FALLBACK_V21, profile.enable_fallback_v21
    )
    if cli_set:
        cli_overrides["enable_fallback_v21"] = enable_fallback_v21
    elif env_set:
        env_overrides[ENV_ENABLE_FALLBACK_V21] = enable_fallback_v21

    enable_failure_case_logger, cli_set, env_set = _resolve_bool(
        args, "enable_failure_case_logger", ENV_ENABLE_FAILURE_CASE_LOGGER, profile.enable_failure_case_logger
    )
    if cli_set:
        cli_overrides["enable_failure_case_logger"] = enable_failure_case_logger
    elif env_set:
        env_overrides[ENV_ENABLE_FAILURE_CASE_LOGGER] = enable_failure_case_logger

    debug_return_intermediate, cli_set, env_set = _resolve_bool(
        args, "debug_return_intermediate", ENV_DEBUG_RETURN_INTERMEDIATE, profile.debug_return_intermediate
    )
    if cli_set:
        cli_overrides["debug_return_intermediate"] = debug_return_intermediate
    elif env_set:
        env_overrides[ENV_DEBUG_RETURN_INTERMEDIATE] = debug_return_intermediate

    model_name, cli_set, env_set = _resolve_str(args, "surface_model", ENV_OPENAI_MODEL, DEFAULT_SURFACE_MODEL)
    if cli_set:
        cli_overrides["surface_model"] = model_name
    elif env_set:
        env_overrides[ENV_OPENAI_MODEL] = model_name

    api_key, cli_set, env_set = _resolve_str(args, "openai_api_key", ENV_OPENAI_API_KEY, None)
    if cli_set:
        cli_overrides["openai_api_key"] = "<redacted>"
    elif env_set:
        env_overrides[ENV_OPENAI_API_KEY] = "<present>"

    base_url, cli_set, env_set = _resolve_str(args, "surface_base_url", ENV_OPENAI_BASE_URL, None)
    if cli_set:
        cli_overrides["surface_base_url"] = base_url
    elif env_set:
        env_overrides[ENV_OPENAI_BASE_URL] = base_url

    timeout_seconds, cli_set, env_set = _resolve_int(
        args, "surface_timeout_seconds", ENV_OPENAI_TIMEOUT_SECONDS, DEFAULT_SURFACE_TIMEOUT_SECONDS
    )
    if cli_set:
        cli_overrides["surface_timeout_seconds"] = timeout_seconds
    elif env_set:
        env_overrides[ENV_OPENAI_TIMEOUT_SECONDS] = timeout_seconds

    max_retries, cli_set, env_set = _resolve_int(args, "surface_max_retries", ENV_OPENAI_MAX_RETRIES, DEFAULT_SURFACE_MAX_RETRIES)
    if cli_set:
        cli_overrides["surface_max_retries"] = max_retries
    elif env_set:
        env_overrides[ENV_OPENAI_MAX_RETRIES] = max_retries

    failure_case_log_path, cli_set, env_set = _resolve_str(args, "failure_case_log_path", ENV_FAILURE_CASE_LOG_PATH, "logs/generation_failures.jsonl")
    if cli_set:
        cli_overrides["failure_case_log_path"] = failure_case_log_path
    elif env_set:
        env_overrides[ENV_FAILURE_CASE_LOG_PATH] = failure_case_log_path

    prompt_path = Path(str(getattr(args, "surface_prompt_path", DEFAULT_SURFACE_PROMPT_PATH) or DEFAULT_SURFACE_PROMPT_PATH))
    rewrite_prompt_path = Path(
        str(getattr(args, "rewrite_prompt_path", DEFAULT_REWRITE_PROMPT_PATH) or DEFAULT_REWRITE_PROMPT_PATH)
    )
    retry_delay_seconds = int(getattr(args, "surface_retry_delay_seconds", DEFAULT_SURFACE_RETRY_DELAY_SECONDS))

    metadata = RuntimeConfigMetadata(
        profile_name=profile_name,
        env_overrides=env_overrides,
        cli_overrides=cli_overrides,
    )
    feature_flags = FeatureFlags(
        enable_generation_chain_v2=enable_generation_chain_v2,
        enable_llm_surface_generation_v3=enable_llm_surface_generation_v3,
        enable_control_checks=enable_control_checks,
        enable_rewrite_v3=enable_rewrite_v3,
        enable_fallback_v21=enable_fallback_v21,
        enable_failure_case_logger=enable_failure_case_logger,
        debug_return_intermediate=debug_return_intermediate,
    )
    surface = SurfaceRuntimeConfig(
        api_key=api_key,
        model_name=str(model_name or DEFAULT_SURFACE_MODEL),
        prompt_path=prompt_path,
        rewrite_prompt_path=rewrite_prompt_path,
        request_timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        base_url=base_url,
    )
    planner = PlannerRuntimeConfig()
    return GenerationRuntimeConfig(
        metadata=metadata,
        feature_flags=feature_flags,
        surface=surface,
        planner=planner,
        failure_case_log_path=Path(str(failure_case_log_path or "logs/generation_failures.jsonl")),
    )
