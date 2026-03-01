#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rag_service import run_guarded_rag_query
from scripts.runtime import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_PROMPT_VERSION,
    DEFAULT_QDRANT_URL,
    DEFAULT_RAG_MODE,
    default_max_answer_chars,
    default_max_context_chars,
    default_max_tool_calls,
    default_min_avg_score,
    default_min_distinct_docs,
    default_min_score,
    default_top_k,
)


def load_cases(path: Path) -> List[Dict]:
    cases = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            case = json.loads(line)
            if "id" not in case or "query" not in case or "expected_decision" not in case:
                raise ValueError(f"Invalid case at {path}:{line_no}")
            case.setdefault("expected_sources", [])
            case.setdefault("lang", None)
            case.setdefault("doc_type", None)
            case.setdefault("query_lang", case.get("lang"))
            case.setdefault("notes", "")
            cases.append(case)
    return cases


def matches_expected_source(src: str, expected_sources: List[str]) -> bool:
    return any(src.startswith(expected) for expected in expected_sources)


def evaluate_case(case: Dict, result: Dict) -> Dict:
    citations = result.get("citations", [])
    top_chunks = result.get("top_chunks", [])
    expected_sources = case["expected_sources"]

    citation_hit = any(matches_expected_source(citation.get("src", ""), expected_sources) for citation in citations)
    retrieval_hit = any(matches_expected_source(chunk.get("src", ""), expected_sources) for chunk in top_chunks)

    decision_match = result["decision"] == case["expected_decision"]
    expected_grounded = case["expected_decision"] == "grounded"
    abstain_match = (result["decision"] == "abstain") == (case["expected_decision"] == "abstain")

    return {
        "id": case["id"],
        "query": case["query"],
        "query_lang": case.get("query_lang"),
        "lang": case.get("lang"),
        "doc_type": case.get("doc_type"),
        "expected_decision": case["expected_decision"],
        "actual_decision": result["decision"],
        "decision_match": decision_match,
        "retrieval_hit": retrieval_hit if expected_sources else result["decision"] == "abstain",
        "citation_hit": citation_hit if expected_sources else result["decision"] == "abstain",
        "abstain_match": abstain_match,
        "grounded_expected": expected_grounded,
        "grounded_returned": result["decision"] == "grounded",
        "latency_ms": result.get("latency_ms", 0.0),
        "top_score": result["assessment"]["top_score"],
        "avg_score": result["assessment"]["avg_score"],
        "reasons": result["assessment"]["reasons"],
        "citations": [citation.get("src") for citation in citations],
        "notes": case.get("notes", ""),
    }


def score_results(results: List[Dict]) -> Dict:
    total = len(results)
    if total == 0:
        return {
            "total_cases": 0,
            "decision_accuracy": 0.0,
            "retrieval_hit_rate": 0.0,
            "citation_hit_rate": 0.0,
            "abstain_accuracy": 0.0,
            "avg_latency_ms": 0.0,
        }

    return {
        "total_cases": total,
        "decision_accuracy": sum(item["decision_match"] for item in results) / total,
        "retrieval_hit_rate": sum(item["retrieval_hit"] for item in results) / total,
        "citation_hit_rate": sum(item["citation_hit"] for item in results) / total,
        "abstain_accuracy": sum(item["abstain_match"] for item in results) / total,
        "avg_latency_ms": mean(item["latency_ms"] for item in results),
    }


def score_by_language(results: List[Dict]) -> Dict[str, Dict]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for item in results:
        grouped[item.get("query_lang") or "unknown"].append(item)
    return {lang: score_results(items) for lang, items in grouped.items()}


def find_failures(results: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    decision_failures = [item for item in results if not item["decision_match"]]
    retrieval_failures = [
        item for item in results if item["grounded_expected"] and (not item["retrieval_hit"] or not item["citation_hit"])
    ]
    return decision_failures, retrieval_failures


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def render_markdown_report(
    *,
    dataset_path: Path,
    args: argparse.Namespace,
    overall: Dict,
    per_language: Dict[str, Dict],
    results: List[Dict],
) -> str:
    decision_failures, retrieval_failures = find_failures(results)

    lines = [
        "# Keraforge Evaluation Report",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Dataset: `{dataset_path}`",
        f"- LLM: `{args.llm}`",
        f"- Mode: `{args.mode}`",
        f"- Collection: `{args.collection}`",
        f"- Device: `{args.device}`",
        f"- Thresholds: `min_score={args.min_score}`, `min_avg_score={args.min_avg_score}`, `min_distinct_docs={args.min_distinct_docs}`",
        "",
        "## Overall",
        "",
        f"- Cases: {overall['total_cases']}",
        f"- Decision accuracy: {overall['decision_accuracy']:.2%}",
        f"- Retrieval hit rate: {overall['retrieval_hit_rate']:.2%}",
        f"- Citation hit rate: {overall['citation_hit_rate']:.2%}",
        f"- Abstain accuracy: {overall['abstain_accuracy']:.2%}",
        f"- Average latency: {overall['avg_latency_ms']:.2f} ms",
        "",
        "## By Query Language",
        "",
        "| Query Lang | Cases | Decision Acc. | Retrieval Hit | Citation Hit | Abstain Acc. | Avg Latency (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for lang, score in sorted(per_language.items()):
        lines.append(
            f"| {lang} | {score['total_cases']} | {score['decision_accuracy']:.2%} | "
            f"{score['retrieval_hit_rate']:.2%} | {score['citation_hit_rate']:.2%} | "
            f"{score['abstain_accuracy']:.2%} | {score['avg_latency_ms']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Failure Modes",
            "",
        ]
    )

    if not decision_failures and not retrieval_failures:
        lines.append("- No failures in the current dataset.")
    else:
        if decision_failures:
            lines.append("- Decision mismatches:")
            for item in decision_failures:
                lines.append(
                    f"  - `{item['id']}` expected `{item['expected_decision']}` but got "
                    f"`{item['actual_decision']}` with reasons `{', '.join(item['reasons'])}`."
                )
        if retrieval_failures:
            lines.append("- Grounded retrieval/citation misses:")
            for item in retrieval_failures:
                lines.append(
                    f"  - `{item['id']}` citations `{', '.join(item['citations']) or '-'}` "
                    f"did not match the expected sources."
                )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This corpus currently contains only Romanian and Polish policy documents.",
            "- English and German cases are negative controls that validate abstain behavior under missing language or doc type filters.",
            "- The harness exercises the same guarded query path used by the CLI and API, including workflow traces and tool-call logic.",
        ]
    )

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Keraforge multilingual RAG evaluation harness.")
    parser.add_argument("--dataset", default="eval/questions.jsonl", help="Path to a JSONL eval dataset.")
    parser.add_argument("--output_json", default="reports/eval_results.json", help="Where to write the JSON results.")
    parser.add_argument("--output_md", default="reports/eval_report.md", help="Where to write the markdown report.")
    parser.add_argument("--qdrant", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top_k", type=int, default=default_top_k())
    parser.add_argument("--llm", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--embed_model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--mode", choices=["workflow", "agent"], default=DEFAULT_RAG_MODE)
    parser.add_argument("--max_tool_calls", type=int, default=default_max_tool_calls())
    parser.add_argument("--min_score", type=float, default=default_min_score())
    parser.add_argument("--min_avg_score", type=float, default=default_min_avg_score())
    parser.add_argument("--min_distinct_docs", type=int, default=default_min_distinct_docs())
    parser.add_argument("--max_context_chars", type=int, default=default_max_context_chars())
    parser.add_argument("--max_answer_chars", type=int, default=default_max_answer_chars())
    parser.add_argument("--prompt_version", default=DEFAULT_PROMPT_VERSION)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    cases = load_cases(dataset_path)
    if args.limit is not None:
        cases = cases[: args.limit]

    results = []
    for case in cases:
        result = run_guarded_rag_query(
            query=case["query"],
            qdrant=args.qdrant,
            collection=args.collection,
            top_k=args.top_k,
            lang=case.get("lang"),
            doc_type=case.get("doc_type"),
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
        )
        case_result = evaluate_case(case, result)
        results.append(case_result)

        print(
            f"[{case_result['id']}] expected={case_result['expected_decision']} "
            f"actual={case_result['actual_decision']} "
            f"retrieval_hit={case_result['retrieval_hit']} "
            f"citation_hit={case_result['citation_hit']} "
            f"latency_ms={case_result['latency_ms']:.2f}"
        )

    overall = score_results(results)
    per_language = score_by_language(results)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "config": {
            "llm": args.llm,
            "mode": args.mode,
            "collection": args.collection,
            "device": args.device,
            "top_k": args.top_k,
            "min_score": args.min_score,
            "min_avg_score": args.min_avg_score,
            "min_distinct_docs": args.min_distinct_docs,
            "max_context_chars": args.max_context_chars,
            "max_answer_chars": args.max_answer_chars,
        },
        "overall": overall,
        "by_query_language": per_language,
        "results": results,
    }

    write_json(Path(args.output_json), payload)
    report = render_markdown_report(
        dataset_path=dataset_path,
        args=args,
        overall=overall,
        per_language=per_language,
        results=results,
    )
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")

    print("")
    print(f"decision_accuracy={overall['decision_accuracy']:.2%}")
    print(f"retrieval_hit_rate={overall['retrieval_hit_rate']:.2%}")
    print(f"citation_hit_rate={overall['citation_hit_rate']:.2%}")
    print(f"abstain_accuracy={overall['abstain_accuracy']:.2%}")
    print(f"avg_latency_ms={overall['avg_latency_ms']:.2f}")
    print(f"wrote_json={args.output_json}")
    print(f"wrote_md={args.output_md}")


if __name__ == "__main__":
    main()
