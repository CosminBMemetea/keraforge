#!/usr/bin/env python3
import json
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional
from uuid import uuid4

from scripts.runtime import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_QDRANT_URL,
    DEFAULT_TRACE_LOG_PATH,
    default_max_answer_chars,
    default_max_context_chars,
    default_min_avg_score,
    default_min_distinct_docs,
    default_min_score,
    default_ollama_model,
    default_sentence_transformer_device,
    default_top_k,
)


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return

    load_dotenv()


def build_filters(lang: Optional[str], doc_type: Optional[str]):
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

    filters = []
    if lang:
        filters.append(MetadataFilter(key="language", value=lang.upper()))
    if doc_type:
        filters.append(MetadataFilter(key="doc_type", value=doc_type.lower()))
    return MetadataFilters(filters=filters) if filters else None


def setup_llm(llm_choice: str):
    llm_choice = llm_choice.lower()

    if llm_choice == "ollama":
        from llama_index.llms.ollama import Ollama

        return Ollama(
            model=default_ollama_model(),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            request_timeout=float(os.getenv("OLLAMA_TIMEOUT", "240")),
        )

    from llama_index.llms.openai import OpenAI
    return OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def node_score(node) -> float:
    return float(node.score or 0.0)


def build_source_ref(meta: Dict) -> str:
    return f"{meta.get('path')}#chunk={meta.get('chunk_index')}"


def format_score(value: float) -> str:
    return f"{value:.4f}"


def classify_query_type(query: str) -> str:
    text = query.strip().lower()
    request_markers = (
        "list",
        "summarize",
        "extract",
        "show",
        "give me",
        "provide",
        "draft",
        "generate",
        "enumera",
        "rezuma",
        "arata",
        "da-mi",
        "listeaza",
        "podaj",
        "pokaz",
        "wypisz",
        "streść",
    )
    if text.endswith("?"):
        return "question"
    if any(text.startswith(marker) for marker in request_markers):
        return "request"
    return "question"


def format_citations(nodes) -> List[Dict]:
    cites = []
    for i, n in enumerate(nodes, 1):
        meta = n.node.metadata or {}
        cites.append(
            {
                "n": i,
                "score": node_score(n),
                "title": meta.get("title"),
                "path": meta.get("path"),
                "chunk_index": meta.get("chunk_index"),
                "src": build_source_ref(meta),
                "lang": meta.get("language"),
                "doc_type": meta.get("doc_type"),
            }
        )
    return cites


def format_top_chunks(nodes) -> List[Dict]:
    chunks = []
    for i, n in enumerate(nodes, 1):
        meta = n.node.metadata or {}
        full_text = n.node.get_content().strip()
        chunks.append(
            {
                "n": i,
                "score": node_score(n),
                "title": meta.get("title"),
                "src": build_source_ref(meta),
                "text": full_text,
                "snippet": full_text.replace("\n", " ")[:240],
            }
        )
    return chunks


def assess_retrieval(nodes, min_score: float, min_avg_score: float, min_distinct_docs: int) -> Dict:
    scores = [node_score(node) for node in nodes]
    docs = {node.node.metadata.get("path") for node in nodes if node.node.metadata.get("path")}
    top_score = max(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0

    reasons = []
    if not nodes:
        reasons.append("no_nodes")
    if top_score < min_score:
        reasons.append(f"top_score<{min_score:.2f}")
    if avg_score < min_avg_score:
        reasons.append(f"avg_score<{min_avg_score:.2f}")
    if len(docs) < min_distinct_docs:
        reasons.append(f"distinct_docs<{min_distinct_docs}")

    return {
        "grounded": not reasons,
        "reasons": reasons or ["ok"],
        "top_score": top_score,
        "avg_score": avg_score,
        "distinct_docs": len(docs),
        "retrieved_nodes": len(nodes),
    }


def build_clarifying_question(lang: Optional[str], doc_type: Optional[str]) -> str:
    lang = (lang or "").upper()
    filter_hint = ""
    if doc_type:
        filter_hint = f" for doc_type '{doc_type.lower()}'"

    if lang == "RO":
        return f"Poti restrange intrebarea la un subiect, document sau versiune mai precisa{filter_hint}?"
    if lang == "PL":
        return f"Czy mozesz doprecyzowac pytanie do konkretnego tematu, dokumentu lub wersji{filter_hint}?"
    return f"Can you narrow the question to a more specific topic, document, or version{filter_hint}?"


def explain_runtime_error(exc: Exception, llm_choice: str) -> str:
    message = str(exc)
    if "insufficient_quota" in message or "Error code: 429" in message:
        return "OpenAI quota is exhausted. Use --llm ollama or update OpenAI billing."
    if llm_choice == "ollama" and "not found" in message:
        model = default_ollama_model()
        return f"Ollama model '{model}' is missing. Run: ollama pull {model}"
    if "Connection refused" in message or "Failed to connect" in message:
        return "Qdrant or Ollama is not reachable. Start the required local service and retry."
    return f"LLM request failed: {message}"


def build_context(nodes, max_context_chars: int) -> Dict:
    context_parts = []
    total = 0
    used_nodes = 0
    for i, n in enumerate(nodes, 1):
        meta = n.node.metadata or {}
        chunk = n.node.get_content().strip()
        header = f"[{i}] {meta.get('title')} | {meta.get('path')}#chunk={meta.get('chunk_index')}\n"
        block = header + chunk + "\n\n"
        if total + len(block) > max_context_chars:
            break
        context_parts.append(block)
        total += len(block)
        used_nodes += 1

    return {
        "text": "".join(context_parts).strip(),
        "chars": total,
        "used_nodes": used_nodes,
    }


def print_guardrail_report(assessment: Dict) -> None:
    print("\n=== RETRIEVAL ASSESSMENT ===\n")
    print(f"decision={'grounded' if assessment['grounded'] else 'abstain'}")
    print(f"top_score={format_score(assessment['top_score'])}")
    print(f"avg_score={format_score(assessment['avg_score'])}")
    print(f"distinct_docs={assessment['distinct_docs']}")
    print(f"retrieved_nodes={assessment['retrieved_nodes']}")
    print(f"reasons={', '.join(assessment['reasons'])}")


def make_step(step: str, status: str, started_at: float, details: Optional[Dict] = None) -> Dict:
    return {
        "step": step,
        "status": status,
        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        "details": details or {},
    }


def append_trace_log(path: str, record: Dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def append_error_trace(
    *,
    trace_log_path: str,
    query_id: str,
    query: str,
    query_type: str,
    prompt_version: str,
    llm: str,
    embed_model: str,
    collection: str,
    workflow_steps: List[Dict],
    request_started: float,
    error: str,
) -> None:
    append_trace_log(
        trace_log_path,
        {
            "query_id": query_id,
            "query": query,
            "query_type": query_type,
            "decision": "error",
            "prompt_version": prompt_version,
            "model": llm,
            "embed_model": embed_model,
            "collection": collection,
            "latency_ms": round((perf_counter() - request_started) * 1000, 2),
            "assessment": None,
            "workflow_steps": workflow_steps,
            "citations": [],
            "error": error,
        },
    )


def build_prompt(query: str, context: str, query_type: str, prompt_version: str) -> str:
    action_line = (
        "Answer the question directly."
        if query_type == "question"
        else "Fulfill the user's request directly using only the context."
    )
    return f"""Prompt version: {prompt_version}
You are a helpful assistant for a multilingual policy knowledge base.
{action_line}
Answer using ONLY the CONTEXT.
If the context is insufficient, say exactly: "I don't know based on the current sources."
Every factual claim must be grounded in the cited chunks.
Keep the answer concise and professional.

CONTEXT:
{context}

QUESTION:
{query}

OUTPUT FORMAT:
- Answer (short, direct)
- Citations: list the bracket numbers you used, e.g. [1], [3]
"""


def apply_generation_guardrails(answer: str, citations: List[Dict], max_answer_chars: int) -> Dict:
    cleaned = answer.strip()
    if len(cleaned) > max_answer_chars:
        cleaned = cleaned[: max_answer_chars - 3].rstrip() + "..."

    citations_line_present = "citations:" in cleaned.lower()
    if not citations_line_present:
        refs = ", ".join(f"[{c['n']}]" for c in citations[:3]) if citations else "[]"
        cleaned = f"{cleaned}\n\nCitations: {refs}"

    return {
        "answer": cleaned,
        "citations_line_added": not citations_line_present,
        "answer_chars": len(cleaned),
    }


@lru_cache(maxsize=8)
def get_index_and_settings(
    qdrant: str,
    collection: str,
    embed_model: str,
    device: str,
    llm: str,
):
    from llama_index.core import Settings, StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    try:
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model, device=device)
    except TypeError:
        Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model)

    Settings.llm = setup_llm(llm)

    client = QdrantClient(url=qdrant)
    vector_store = QdrantVectorStore(client=client, collection_name=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    return index, Settings


def run_guarded_rag_query(
    query: str,
    qdrant: str = DEFAULT_QDRANT_URL,
    collection: str = DEFAULT_COLLECTION,
    top_k: int = default_top_k(),
    lang: Optional[str] = None,
    doc_type: Optional[str] = None,
    llm: str = "ollama",
    embed_model: str = DEFAULT_EMBED_MODEL,
    device: str = "auto",
    min_score: float = default_min_score(),
    min_avg_score: float = default_min_avg_score(),
    min_distinct_docs: int = default_min_distinct_docs(),
    max_context_chars: int = default_max_context_chars(),
    max_answer_chars: int = default_max_answer_chars(),
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    trace_log_path: str = DEFAULT_TRACE_LOG_PATH,
) -> Dict:
    query_id = str(uuid4())
    workflow_steps = []
    request_started = perf_counter()
    load_env()
    step_started = perf_counter()
    query_type = classify_query_type(query)
    workflow_steps.append(
        make_step("classify_query", "completed", step_started, {"query_type": query_type})
    )

    try:
        import llama_index
        import qdrant_client
    except ModuleNotFoundError as exc:
        message = (
            "Missing RAG dependencies. Install: "
            "pip install -U python-dotenv llama-index llama-index-vector-stores-qdrant "
            "llama-index-embeddings-huggingface llama-index-llms-ollama fastapi uvicorn"
        )
        append_error_trace(
            trace_log_path=trace_log_path,
            query_id=query_id,
            query=query,
            query_type=query_type,
            prompt_version=prompt_version,
            llm=llm,
            embed_model=embed_model,
            collection=collection,
            workflow_steps=workflow_steps,
            request_started=request_started,
            error=message,
        )
        raise RuntimeError(message) from exc

    resolved_device = default_sentence_transformer_device(device)

    try:
        step_started = perf_counter()
        index, settings = get_index_and_settings(
            qdrant=qdrant,
            collection=collection,
            embed_model=embed_model,
            device=resolved_device,
            llm=llm,
        )
        filters = build_filters(lang, doc_type)
        retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
        nodes = retriever.retrieve(query)
        workflow_steps.append(
            make_step(
                "retrieve",
                "completed",
                step_started,
                {
                    "top_k": top_k,
                    "lang": lang.upper() if lang else None,
                    "doc_type": doc_type.lower() if doc_type else None,
                    "retrieved_nodes": len(nodes),
                },
            )
        )
    except Exception as exc:
        message = explain_runtime_error(exc, llm)
        append_error_trace(
            trace_log_path=trace_log_path,
            query_id=query_id,
            query=query,
            query_type=query_type,
            prompt_version=prompt_version,
            llm=llm,
            embed_model=embed_model,
            collection=collection,
            workflow_steps=workflow_steps,
            request_started=request_started,
            error=message,
        )
        raise RuntimeError(message) from exc

    step_started = perf_counter()
    assessment = assess_retrieval(
        nodes,
        min_score=min_score,
        min_avg_score=min_avg_score,
        min_distinct_docs=min_distinct_docs,
    )
    citations = format_citations(nodes)
    top_chunks = format_top_chunks(nodes)
    context = build_context(nodes, max_context_chars=max_context_chars)

    if not context["text"]:
        assessment["grounded"] = False
        assessment["reasons"] = assessment["reasons"] + ["empty_context_budget"]

    workflow_steps.append(
        make_step(
            "guardrail_assessment",
            "completed",
            step_started,
            {
                "decision": "grounded" if assessment["grounded"] else "abstain",
                "top_score": assessment["top_score"],
                "avg_score": assessment["avg_score"],
                "distinct_docs": assessment["distinct_docs"],
                "reasons": assessment["reasons"],
            },
        )
    )

    result = {
        "query_id": query_id,
        "query": query,
        "query_type": query_type,
        "prompt_version": prompt_version,
        "decision": "grounded" if assessment["grounded"] else "abstain",
        "answer": "I don't know based on the current sources.",
        "clarifying_question": None,
        "assessment": assessment,
        "citations": citations,
        "top_chunks": top_chunks,
        "context_chars": context["chars"],
        "context_nodes_used": context["used_nodes"],
        "lang": lang.upper() if lang else None,
        "doc_type": doc_type.lower() if doc_type else None,
        "llm": llm,
        "embed_model": embed_model,
        "device": resolved_device,
        "collection": collection,
        "workflow_steps": workflow_steps,
    }

    if not assessment["grounded"]:
        result["clarifying_question"] = build_clarifying_question(lang, doc_type)
        result["latency_ms"] = round((perf_counter() - request_started) * 1000, 2)
        append_trace_log(
            trace_log_path,
            {
                "query_id": query_id,
                "query": query,
                "query_type": query_type,
                "decision": result["decision"],
                "prompt_version": prompt_version,
                "model": llm,
                "embed_model": embed_model,
                "collection": collection,
                "latency_ms": result["latency_ms"],
                "assessment": assessment,
                "workflow_steps": workflow_steps,
                "citations": citations,
                "error": None,
            },
        )
        return result

    prompt = build_prompt(query, context["text"], query_type=query_type, prompt_version=prompt_version)

    try:
        step_started = perf_counter()
        raw_answer = settings.llm.complete(prompt).text.strip()
        workflow_steps.append(
            make_step(
                "generate_answer",
                "completed",
                step_started,
                {"prompt_chars": len(prompt), "context_chars": context["chars"]},
            )
        )
    except Exception as exc:
        message = explain_runtime_error(exc, llm)
        append_error_trace(
            trace_log_path=trace_log_path,
            query_id=query_id,
            query=query,
            query_type=query_type,
            prompt_version=prompt_version,
            llm=llm,
            embed_model=embed_model,
            collection=collection,
            workflow_steps=workflow_steps,
            request_started=request_started,
            error=message,
        )
        raise RuntimeError(message) from exc

    step_started = perf_counter()
    post_guardrails = apply_generation_guardrails(raw_answer, citations, max_answer_chars=max_answer_chars)
    result["answer"] = post_guardrails["answer"]
    workflow_steps.append(
        make_step(
            "apply_guardrails",
            "completed",
            step_started,
            {
                "answer_chars": post_guardrails["answer_chars"],
                "citations_line_added": post_guardrails["citations_line_added"],
                "max_answer_chars": max_answer_chars,
            },
        )
    )
    result["latency_ms"] = round((perf_counter() - request_started) * 1000, 2)

    append_trace_log(
        trace_log_path,
        {
            "query_id": query_id,
            "query": query,
            "query_type": query_type,
            "decision": result["decision"],
            "prompt_version": prompt_version,
            "model": llm,
            "embed_model": embed_model,
            "collection": collection,
            "latency_ms": result["latency_ms"],
            "assessment": assessment,
            "workflow_steps": workflow_steps,
            "citations": citations,
            "answer_preview": result["answer"][:240],
            "error": None,
        },
    )
    return result
