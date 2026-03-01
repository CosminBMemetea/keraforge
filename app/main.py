from time import perf_counter
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from scripts.rag_service import run_guarded_rag_query
from scripts.runtime import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_QDRANT_URL,
    default_max_context_chars,
    default_min_avg_score,
    default_min_distinct_docs,
    default_min_score,
    default_top_k,
)


class CitationOut(BaseModel):
    n: int
    score: float
    title: Optional[str] = None
    path: Optional[str] = None
    chunk_index: Optional[int] = None
    src: str
    lang: Optional[str] = None
    doc_type: Optional[str] = None


class ChunkOut(BaseModel):
    n: int
    score: float
    title: Optional[str] = None
    src: str
    text: str
    snippet: str


class AssessmentOut(BaseModel):
    grounded: bool
    reasons: List[str]
    top_score: float
    avg_score: float
    distinct_docs: int
    retrieved_nodes: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    qdrant: str = DEFAULT_QDRANT_URL
    collection: str = DEFAULT_COLLECTION
    top_k: int = Field(default_factory=default_top_k, ge=1, le=10)
    lang: Optional[str] = None
    doc_type: Optional[str] = None
    llm: str = Field(default="ollama", pattern="^(ollama|openai)$")
    embed_model: str = DEFAULT_EMBED_MODEL
    device: str = Field(default="auto", pattern="^(auto|cpu|mps|cuda)$")
    min_score: float = Field(default_factory=default_min_score, ge=0.0, le=1.0)
    min_avg_score: float = Field(default_factory=default_min_avg_score, ge=0.0, le=1.0)
    min_distinct_docs: int = Field(default_factory=default_min_distinct_docs, ge=1, le=10)
    max_context_chars: int = Field(default_factory=default_max_context_chars, ge=200, le=12000)


class QueryResponse(BaseModel):
    query: str
    decision: str
    answer: str
    clarifying_question: Optional[str] = None
    assessment: AssessmentOut
    citations: List[CitationOut]
    top_chunks: List[ChunkOut]
    context_chars: int
    context_nodes_used: int
    lang: Optional[str] = None
    doc_type: Optional[str] = None
    llm: str
    embed_model: str
    device: str
    collection: str
    latency_ms: float


app = FastAPI(
    title="KeraForge RAG API",
    version="0.1.0",
    description="Guarded multilingual RAG API over Qdrant with citations and abstain behavior.",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KeraForge RAG Console</title>
  <style>
    :root {
      --bg: #f3efe6;
      --panel: rgba(255, 251, 245, 0.84);
      --ink: #1d1a16;
      --muted: #756b62;
      --line: rgba(29, 26, 22, 0.12);
      --accent: #c95d3a;
      --accent-2: #254f6b;
      --shadow: 0 24px 60px rgba(31, 24, 18, 0.14);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(201, 93, 58, 0.18), transparent 30%),
        radial-gradient(circle at bottom right, rgba(37, 79, 107, 0.15), transparent 30%),
        linear-gradient(135deg, #f6f2ea, #e9e0d3);
      padding: 32px 18px;
    }
    .shell {
      max-width: 980px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .hero, .panel {
      background: var(--panel);
      backdrop-filter: blur(16px);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 28px;
    }
    .eyebrow {
      letter-spacing: 0.14em;
      text-transform: uppercase;
      font-size: 12px;
      color: var(--accent-2);
      margin-bottom: 10px;
    }
    h1 {
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 0.94;
    }
    p {
      margin: 0;
      max-width: 60ch;
      color: var(--muted);
      font-size: 15px;
      line-height: 1.5;
    }
    form {
      display: grid;
      gap: 14px;
      padding: 24px;
    }
    .row {
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }
    label {
      display: grid;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    input, select, textarea, button {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      font: inherit;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.7);
    }
    textarea {
      min-height: 110px;
      resize: vertical;
    }
    button {
      cursor: pointer;
      border: none;
      background: linear-gradient(135deg, var(--accent), #e08c3d);
      color: white;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .panel {
      padding: 24px;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "SFMono-Regular", "Menlo", monospace;
      font-size: 13px;
      line-height: 1.55;
    }
    .muted { color: var(--muted); }
    @media (max-width: 760px) {
      .row { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 520px) {
      .row { grid-template-columns: 1fr; }
      body { padding: 16px; }
      .hero, .panel { border-radius: 20px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">API + Console</div>
      <h1>KeraForge RAG Console</h1>
      <p>Query the multilingual policy index through the same guarded retrieval path used by the CLI. Weak retrieval abstains instead of improvising.</p>
    </section>

    <section class="panel">
      <form id="query-form">
        <label>
          Query
          <textarea id="query" name="query">politica de date sintetice</textarea>
        </label>
        <div class="row">
          <label>
            Language
            <select id="lang" name="lang">
              <option value="">Any</option>
              <option value="RO" selected>RO</option>
              <option value="PL">PL</option>
              <option value="EN">EN</option>
              <option value="DE">DE</option>
            </select>
          </label>
          <label>
            LLM
            <select id="llm" name="llm">
              <option value="ollama" selected>ollama</option>
              <option value="openai">openai</option>
            </select>
          </label>
          <label>
            Top K
            <input id="top_k" name="top_k" type="number" min="1" max="10" value="3">
          </label>
          <label>
            Min score
            <input id="min_score" name="min_score" type="number" min="0" max="1" step="0.01" value="0.35">
          </label>
        </div>
        <button type="submit">Run Guarded Query</button>
      </form>
    </section>

    <section class="panel">
      <div class="muted" style="margin-bottom: 12px;">Response</div>
      <pre id="output">Submit a query to call POST /query.</pre>
    </section>
  </div>

  <script>
    const form = document.getElementById("query-form");
    const output = document.getElementById("output");

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      output.textContent = "Running query...";
      const payload = {
        query: document.getElementById("query").value,
        lang: document.getElementById("lang").value || null,
        llm: document.getElementById("llm").value,
        top_k: Number(document.getElementById("top_k").value),
        min_score: Number(document.getElementById("min_score").value)
      };

      const response = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await response.json();
      output.textContent = JSON.stringify(data, null, 2);
    });
  </script>
</body>
</html>"""


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    started = perf_counter()
    try:
        result = run_guarded_rag_query(**payload.model_dump())
    except RuntimeError as exc:
        message = str(exc)
        status_code = 503 if any(
            token in message.lower()
            for token in ["not reachable", "missing", "quota", "failed"]
        ) else 500
        raise HTTPException(status_code=status_code, detail=message) from exc

    latency_ms = round((perf_counter() - started) * 1000, 2)
    return QueryResponse(**result, latency_ms=latency_ms)
