# Keraforge Evaluation Report

- Generated: 2026-03-01T16:09:55.447088+00:00
- Dataset: `eval/questions.jsonl`
- LLM: `ollama`
- Mode: `workflow`
- Collection: `querra_docs`
- Device: `auto`
- Thresholds: `min_score=0.35`, `min_avg_score=0.25`, `min_distinct_docs=1`

## Overall

- Cases: 12
- Decision accuracy: 100.00%
- Retrieval hit rate: 100.00%
- Citation hit rate: 100.00%
- Abstain accuracy: 100.00%
- Average latency: 2749.57 ms

## By Query Language

| Query Lang | Cases | Decision Acc. | Retrieval Hit | Citation Hit | Abstain Acc. | Avg Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DE | 2 | 100.00% | 100.00% | 100.00% | 100.00% | 797.03 |
| EN | 2 | 100.00% | 100.00% | 100.00% | 100.00% | 959.06 |
| PL | 4 | 100.00% | 100.00% | 100.00% | 100.00% | 1549.44 |
| RO | 4 | 100.00% | 100.00% | 100.00% | 100.00% | 5821.23 |

## Failure Modes

- No failures in the current dataset.

## Notes

- This corpus currently contains only Romanian and Polish policy documents.
- English and German cases are negative controls that validate abstain behavior under missing language or doc type filters.
- The harness exercises the same guarded query path used by the CLI and API, including workflow traces and tool-call logic.
