### Run Qdrant
docker compose up -d

### Ingest docs
source .venv/bin/activate
python scripts/ingest.py --docs docs
python scripts/ingest.py --docs docs --device mps

### Semantic search (with filters)
python scripts/search.py "politica de date sintetice" --lang RO
python scripts/search.py "zgodność dane syntetyczne" --lang PL


### Day II
pip install -U llama-index llama-index-vector-stores-qdrant llama-index-embeddings-huggingface


set -a
source .env
set +a

ollama pull qwen2.5:1.5b-instruct
pip install -U llama-index-llms-ollama
curl -s http://localhost:11434/api/tags | head

python scripts/rag_query.py "politica de date sintetice" --lang RO --llm ollama --top_k 3

### Day III
python scripts/rag_query.py "politica de date sintetice" --lang RO --llm ollama --top_k 2 --max_context_chars 900
# If retrieval is weak, the script abstains and asks a clarifying question.

### Day IV
uvicorn app.main:app --reload
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"politica de date sintetice","lang":"RO","llm":"ollama","top_k":2,"max_context_chars":900}'

### Day V
python scripts/rag_query.py "politica de date sintetice" --lang RO --llm ollama
tail -n 1 data/logs/rag_traces.jsonl

### Day VI
python scripts/rag_query.py "politica de date sintetice" --lang RO --llm ollama --mode workflow
python scripts/rag_query.py "politica de date sintetice" --lang RO --llm ollama --mode agent

### Day VII
python scripts/eval.py --llm ollama --mode workflow
cat reports/eval_report.md

# Current local eval snapshot
# - 12 cases
# - 100% decision accuracy
# - 100% retrieval hit rate
# - 100% citation hit rate
# - 100% abstain accuracy
