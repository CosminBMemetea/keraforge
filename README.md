# keraforge

# Run qdrant locally

```bash
docker compose up -d
docker ps
curl -s http://localhost:6333/ | head # SANITY CHECK
```
To set up your environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

pip install qdrant-client sentence-transformers fastapi uvicorn python-frontmatter rich
```