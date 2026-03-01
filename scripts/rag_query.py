#!/usr/bin/env python3
import argparse
import os
from typing import Dict, List, Optional

from runtime import (
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


def print_guardrail_report(assessment: Dict) -> None:
    print("\n=== RETRIEVAL ASSESSMENT ===\n")
    print(f"decision={'grounded' if assessment['grounded'] else 'abstain'}")
    print(f"top_score={format_score(assessment['top_score'])}")
    print(f"avg_score={format_score(assessment['avg_score'])}")
    print(f"distinct_docs={assessment['distinct_docs']}")
    print(f"retrieved_nodes={assessment['retrieved_nodes']}")
    print(f"reasons={', '.join(assessment['reasons'])}")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="User query")
    ap.add_argument("--qdrant", default=DEFAULT_QDRANT_URL, help="Qdrant URL")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION, help="Qdrant collection name")
    ap.add_argument("--top_k", type=int, default=default_top_k())
    ap.add_argument("--lang", default=None, help="Language filter e.g. RO/PL/EN/DE")
    ap.add_argument("--doc_type", default=None, help="doc_type filter e.g. policy/guideline/example")
    ap.add_argument("--llm", default="ollama", choices=["ollama", "openai"])
    ap.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument(
        "--min_score",
        type=float,
        default=default_min_score(),
        help="Minimum top retrieval score required before generation",
    )
    ap.add_argument(
        "--min_avg_score",
        type=float,
        default=default_min_avg_score(),
        help="Minimum average retrieval score required before generation",
    )
    ap.add_argument(
        "--min_distinct_docs",
        type=int,
        default=default_min_distinct_docs(),
        help="Minimum number of distinct source documents required before generation",
    )
    ap.add_argument(
        "--max_context_chars",
        type=int,
        default=default_max_context_chars(),
        help="Trim retrieved context to avoid huge prompts",
    )
    args = ap.parse_args()

    load_env()

    try:
        from llama_index.core import Settings, StorageContext, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing RAG dependencies. Install: "
            "pip install -U python-dotenv llama-index llama-index-vector-stores-qdrant "
            "llama-index-embeddings-huggingface llama-index-llms-ollama"
        ) from exc

    device = default_sentence_transformer_device(args.device)

    try:
        Settings.embed_model = HuggingFaceEmbedding(model_name=args.embed_model, device=device)
    except TypeError:
        Settings.embed_model = HuggingFaceEmbedding(model_name=args.embed_model)

    Settings.llm = setup_llm(args.llm)

    client = QdrantClient(url=args.qdrant)
    vector_store = QdrantVectorStore(client=client, collection_name=args.collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)

    filters = build_filters(args.lang, args.doc_type)
    try:
        retriever = index.as_retriever(similarity_top_k=args.top_k, filters=filters)
        nodes = retriever.retrieve(args.query)
    except Exception as exc:
        raise SystemExit(explain_runtime_error(exc, args.llm)) from exc

    assessment = assess_retrieval(
        nodes,
        min_score=args.min_score,
        min_avg_score=args.min_avg_score,
        min_distinct_docs=args.min_distinct_docs,
    )

    citations = format_citations(nodes)

    context_parts = []
    total = 0
    for i, n in enumerate(nodes, 1):
        meta = n.node.metadata or {}
        chunk = n.node.get_content().strip()
        header = f"[{i}] {meta.get('title')} | {meta.get('path')}#chunk={meta.get('chunk_index')}\n"
        block = header + chunk + "\n\n"
        if total + len(block) > args.max_context_chars:
            break
        context_parts.append(block)
        total += len(block)

    context = "".join(context_parts).strip()
    if not context:
        assessment["grounded"] = False
        assessment["reasons"] = assessment["reasons"] + ["empty_context_budget"]

    print_guardrail_report(assessment)

    if not assessment["grounded"]:
        print("\n=== ANSWER ===\n")
        print("I don't know based on the current sources.")
        print(f"Clarifying question: {build_clarifying_question(args.lang, args.doc_type)}")

        print("\n=== CITATIONS (retrieved sources) ===\n")
        if citations:
            for c in citations:
                print(
                    f"[{c['n']}] score={format_score(c['score'])} doc={c['title']} "
                    f"chunk={c['chunk_index']} :: {c['src']} (lang={c['lang']} type={c['doc_type']})"
                )
        else:
            print("No retrieved sources.")
        return

    prompt = f"""You are a helpful assistant.
Answer using ONLY the CONTEXT.
If the context is insufficient, say exactly: "I don't know based on the current sources."
Every factual claim must be grounded in the cited chunks.

CONTEXT:
{context}

QUESTION:
{args.query}

OUTPUT FORMAT:
- Answer (short, direct)
- Citations: list the bracket numbers you used, e.g. [1], [3]
"""

    try:
        answer = Settings.llm.complete(prompt).text.strip()
    except Exception as exc:
        raise SystemExit(explain_runtime_error(exc, args.llm)) from exc

    print("\n=== ANSWER ===\n")
    print(answer)

    print("\n=== CITATIONS (retrieved sources) ===\n")
    for c in citations:
        print(
            f"[{c['n']}] score={format_score(c['score'])} doc={c['title']} "
            f"chunk={c['chunk_index']} :: {c['src']} (lang={c['lang']} type={c['doc_type']})"
        )

    print("\n=== TOP CHUNKS (snippets) ===\n")
    for i, n in enumerate(nodes, 1):
        meta = n.node.metadata or {}
        snippet = n.node.get_content().replace("\n", " ")[:240]
        print(f"[{i}] score={n.score:.4f} {meta.get('path')}#chunk={meta.get('chunk_index')} :: {snippet}...")


if __name__ == "__main__":
    main()
