#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rag_service import print_guardrail_report, run_guarded_rag_query
from scripts.runtime import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_QDRANT_URL,
    DEFAULT_RAG_MODE,
    DEFAULT_TRACE_LOG_PATH,
    default_max_answer_chars,
    default_max_context_chars,
    default_max_tool_calls,
    default_min_avg_score,
    default_min_distinct_docs,
    default_min_score,
    default_top_k,
)


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
    ap.add_argument("--mode", default=DEFAULT_RAG_MODE, choices=["workflow", "agent"])
    ap.add_argument("--max_tool_calls", type=int, default=default_max_tool_calls())
    ap.add_argument("--prompt_version", default=DEFAULT_PROMPT_VERSION)
    ap.add_argument("--trace_log_path", default=DEFAULT_TRACE_LOG_PATH)
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
    ap.add_argument(
        "--max_answer_chars",
        type=int,
        default=default_max_answer_chars(),
        help="Trim generated answer length after guardrails",
    )
    args = ap.parse_args()

    try:
        result = run_guarded_rag_query(
            query=args.query,
            qdrant=args.qdrant,
            collection=args.collection,
            top_k=args.top_k,
            lang=args.lang,
            doc_type=args.doc_type,
            llm=args.llm,
            embed_model=args.embed_model,
            device=args.device,
            mode=args.mode,
            max_tool_calls=args.max_tool_calls,
            min_score=args.min_score,
            min_avg_score=args.min_avg_score,
            min_distinct_docs=args.min_distinct_docs,
            max_context_chars=args.max_context_chars,
            max_answer_chars=args.max_answer_chars,
            prompt_version=args.prompt_version,
            trace_log_path=args.trace_log_path,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    print("\n=== WORKFLOW ===\n")
    print(f"query_id={result['query_id']}")
    print(f"query_type={result['query_type']}")
    print(f"mode={result['mode']}")
    print(f"prompt_version={result['prompt_version']}")
    print(f"latency_ms={result['latency_ms']:.2f}")
    for step in result["workflow_steps"]:
        print(
            f"- {step['step']} status={step['status']} duration_ms={step['duration_ms']:.2f} "
            f"details={step['details']}"
        )

    print("\n=== TOOLS ===\n")
    if result["tool_calls"]:
        for tool_call in result["tool_calls"]:
            print(
                f"- {tool_call['tool']} status={tool_call['status']} "
                f"duration_ms={tool_call['duration_ms']:.2f} "
                f"inputs={tool_call['inputs']} outputs={tool_call['outputs']}"
            )
    else:
        print("No tool calls recorded.")

    print_guardrail_report(result["assessment"])

    if result["decision"] != "grounded":
        print("\n=== ANSWER ===\n")
        print(result["answer"])
        print(f"Clarifying question: {result['clarifying_question']}")

        print("\n=== CITATIONS (retrieved sources) ===\n")
        if result["citations"]:
            for c in result["citations"]:
                print(
                    f"[{c['n']}] score={c['score']:.4f} doc={c['title']} "
                    f"chunk={c['chunk_index']} :: {c['src']} (lang={c['lang']} type={c['doc_type']})"
                )
        else:
            print("No retrieved sources.")
        return

    print("\n=== ANSWER ===\n")
    print(result["answer"])

    print("\n=== CITATIONS (retrieved sources) ===\n")
    for c in result["citations"]:
        print(
            f"[{c['n']}] score={c['score']:.4f} doc={c['title']} "
            f"chunk={c['chunk_index']} :: {c['src']} (lang={c['lang']} type={c['doc_type']})"
        )

    print("\n=== TOP CHUNKS (snippets) ===\n")
    for chunk in result["top_chunks"]:
        print(f"[{chunk['n']}] score={chunk['score']:.4f} {chunk['src']} :: {chunk['snippet']}...")


if __name__ == "__main__":
    main()
