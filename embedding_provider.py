#!/usr/bin/env python
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
SUPPORTED_EMBEDDING_PROVIDERS = {"local", "openai"}


class EmbeddingConfigError(Exception):
    """Raised when the embedding provider configuration is invalid."""


class EmbeddingRuntimeError(Exception):
    """Raised when embedding generation fails at runtime."""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = DEFAULT_EMBEDDING_PROVIDER
    model_name: str | None = None

    @property
    def resolved_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        if self.provider == "local":
            return DEFAULT_LOCAL_MODEL
        if self.provider == "openai":
            return DEFAULT_OPENAI_MODEL
        raise EmbeddingConfigError(
            f"Unsupported embedding provider: {self.provider}. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))}"
        )


def validate_embedding_provider(provider: str) -> None:
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise EmbeddingConfigError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: {', '.join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))}"
        )


@lru_cache(maxsize=4)
def _get_local_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbeddingConfigError(
            "Missing dependency 'sentence-transformers'.\n"
            "Please run:\n"
            'python -m pip install -r requirements.txt'
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EmbeddingRuntimeError(
            f"Failed to load local embedding model '{model_name}'.\n"
            "If this is the first run, model weights will be downloaded automatically, which is normal.\n"
            f"Original error: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EmbeddingConfigError(
            "You selected '--embedding-provider openai' but OPENAI_API_KEY is not set.\n"
            "Windows PowerShell example:\n"
            '$env:OPENAI_API_KEY = "your-openai-api-key"'
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise EmbeddingConfigError(
            "Missing dependency 'openai'.\n"
            "Please run:\n"
            'python -m pip install -r requirements.txt'
        ) from exc

    return OpenAI(api_key=api_key)


def _embed_local(texts: Sequence[str], model_name: str) -> list[list[float]]:
    model = _get_local_model(model_name)

    try:
        embeddings = model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise EmbeddingRuntimeError(
            f"Local embedding generation failed for model '{model_name}'.\n"
            f"Original error: {exc}"
        ) from exc

    return embeddings.tolist()


def _embed_openai(texts: Sequence[str], model_name: str) -> list[list[float]]:
    client = _get_openai_client()

    try:
        response = client.embeddings.create(model=model_name, input=list(texts))
    except Exception as exc:
        message = str(exc)
        if "insufficient_quota" in message or "429" in message:
            raise EmbeddingRuntimeError(
                "OpenAI embeddings request failed because the API key has no available quota "
                "or the quota has been exhausted.\n"
                "You can switch to local embeddings with:\n"
                "--embedding-provider local"
            ) from exc
        raise EmbeddingRuntimeError(
            "OpenAI embeddings request failed.\n"
            "Please check your API key, billing/quota, network access, or switch to local embeddings.\n"
            f"Original error: {message}"
        ) from exc

    return [item.embedding for item in response.data]


def embed_texts(texts: Sequence[str], provider: str, model_name: str | None = None) -> list[list[float]]:
    validate_embedding_provider(provider)
    config = EmbeddingConfig(provider=provider, model_name=model_name)

    if not texts:
        return []

    if config.provider == "local":
        return _embed_local(texts, config.resolved_model_name)
    return _embed_openai(texts, config.resolved_model_name)
