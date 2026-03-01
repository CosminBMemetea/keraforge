#!/usr/bin/env python3
import os
from functools import lru_cache
from typing import Dict, List, Optional

from scripts.runtime import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_QDRANT_URL,
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
) -> Dict:
    load_env()

    try:
        import llama_index
        import qdrant_client
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing RAG dependencies. Install: "
            "pip install -U python-dotenv llama-index llama-index-vector-stores-qdrant "
            "llama-index-embeddings-huggingface llama-index-llms-ollama fastapi uvicorn"
        ) from exc

    resolved_device = default_sentence_transformer_device(device)

    try:
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
    except Exception as exc:
        raise RuntimeError(explain_runtime_error(exc, llm)) from exc

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

    result = {
        "query": query,
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
    }

    if not assessment["grounded"]:
        result["clarifying_question"] = build_clarifying_question(lang, doc_type)
        return result

    prompt = f"""You are a helpful assistant.
Answer using ONLY the CONTEXT.
If the context is insufficient, say exactly: "I don't know based on the current sources."
Every factual claim must be grounded in the cited chunks.

CONTEXT:
{context['text']}

QUESTION:
{query}

OUTPUT FORMAT:
- Answer (short, direct)
- Citations: list the bracket numbers you used, e.g. [1], [3]
"""

    try:
        result["answer"] = settings.llm.complete(prompt).text.strip()
    except Exception as exc:
        raise RuntimeError(explain_runtime_error(exc, llm)) from exc

    return result
