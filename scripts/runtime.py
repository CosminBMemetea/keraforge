#!/usr/bin/env python3
import os
import platform


DEFAULT_COLLECTION = os.getenv("KERAFORGE_COLLECTION", "querra_docs")
DEFAULT_QDRANT_URL = os.getenv("KERAFORGE_QDRANT_URL", "http://localhost:6333")
DEFAULT_EMBED_MODEL = os.getenv(
    "KERAFORGE_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
DEFAULT_PROMPT_VERSION = os.getenv("KERAFORGE_PROMPT_VERSION", "rag-v1")
DEFAULT_TRACE_LOG_PATH = os.getenv("KERAFORGE_TRACE_LOG_PATH", "data/logs/rag_traces.jsonl")


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def default_chunk_size() -> int:
    return int(os.getenv("KERAFORGE_CHUNK_SIZE", "700" if is_apple_silicon() else "900"))


def default_overlap() -> int:
    return int(os.getenv("KERAFORGE_CHUNK_OVERLAP", "100" if is_apple_silicon() else "120"))


def default_embed_batch_size() -> int:
    return int(os.getenv("KERAFORGE_EMBED_BATCH_SIZE", "8" if is_apple_silicon() else "32"))


def default_upsert_batch_size() -> int:
    return int(os.getenv("KERAFORGE_UPSERT_BATCH_SIZE", "64" if is_apple_silicon() else "128"))


def default_top_k() -> int:
    return int(os.getenv("KERAFORGE_TOP_K", "3" if is_apple_silicon() else "5"))


def default_max_context_chars() -> int:
    return int(os.getenv("KERAFORGE_MAX_CONTEXT_CHARS", "2500" if is_apple_silicon() else "5000"))


def default_max_answer_chars() -> int:
    return int(os.getenv("KERAFORGE_MAX_ANSWER_CHARS", "700"))


def default_min_score() -> float:
    return float(os.getenv("KERAFORGE_MIN_SCORE", "0.35"))


def default_min_avg_score() -> float:
    return float(os.getenv("KERAFORGE_MIN_AVG_SCORE", "0.25"))


def default_min_distinct_docs() -> int:
    return int(os.getenv("KERAFORGE_MIN_DISTINCT_DOCS", "1"))


def default_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct" if is_apple_silicon() else "llama3")


def default_sentence_transformer_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested

    try:
        import torch
    except Exception:
        return "cpu"

    if is_apple_silicon() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"
