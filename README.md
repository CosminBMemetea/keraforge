# keraforge

## Run qdrant locally

```bash
docker compose up -d
docker ps
curl -s http://localhost:6333/ | head # SANITY CHECK
```

## Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install qdrant-client sentence-transformers python-frontmatter python-dotenv \
  llama-index llama-index-vector-stores-qdrant llama-index-embeddings-huggingface \
  llama-index-llms-ollama
```

Copy the local config template and set your own keys if you need OpenAI:

```bash
cp .env.example .env
```

For a MacBook Air/Pro with Apple Silicon and 8 GB RAM, use a small Ollama model:

```bash
ollama pull qwen2.5:1.5b-instruct
```

## Ingest docs

```bash
source .venv/bin/activate

python scripts/ingest.py --docs docs
python scripts/ingest.py --docs docs --device mps
python scripts/ingest.py --docs docs --device cpu --batch_size 4
```

The defaults are tuned down automatically on Apple Silicon:

- smaller chunk size
- smaller embedding batch size
- smaller Qdrant upsert batches
- smaller RAG retrieval/context defaults
- default Ollama model fallback: `qwen2.5:1.5b-instruct`

## Semantic search

```bash
python scripts/search.py "politica de date sintetice" --lang RO
python scripts/search.py "zasady zgodności danych" --lang PL
python scripts/search.py "compliance synthetic datasets" --lang EN
```

If `mps` is unstable on your machine, force CPU:

```bash
python scripts/search.py "politica de date sintetice" --lang RO --device cpu
```

## RAG query

```bash
source .venv/bin/activate
python scripts/rag_query.py "politica de date sintetice" --lang RO --llm ollama
python scripts/rag_query.py "zgodność dane syntetyczne" --lang PL --llm ollama --device cpu
```

Day 3 guardrails are built into `scripts/rag_query.py`:

- explicit retrieval assessment before generation
- abstain behavior when retrieval is weak
- clarifying follow-up question on abstain
- provenance output with document name and chunk id

You can tune the trust thresholds from the CLI:

```bash
python scripts/rag_query.py "politica de date sintetice" --lang RO --llm ollama \
  --top_k 2 --max_context_chars 900 --min_score 0.35 --min_avg_score 0.25
```

## Troubleshooting

```bash
pip show qdrant-client
pip install -U qdrant-client
```

`scripts/search.py` already handles both older `search()` and newer `query_points()` Qdrant client APIs.

If Ollama says the model is missing, pull it first:

```bash
ollama pull qwen2.5:1.5b-instruct
```
