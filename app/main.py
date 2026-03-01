from time import perf_counter
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from scripts.rag_service import run_guarded_rag_query
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


class WorkflowStepOut(BaseModel):
    step: str
    status: str
    duration_ms: float
    details: dict


class ToolCallOut(BaseModel):
    tool: str
    status: str
    duration_ms: float
    inputs: dict
    outputs: dict


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
    mode: str = Field(default=DEFAULT_RAG_MODE, pattern="^(workflow|agent)$")
    max_tool_calls: int = Field(default_factory=default_max_tool_calls, ge=1, le=4)
    prompt_version: str = DEFAULT_PROMPT_VERSION
    trace_log_path: str = DEFAULT_TRACE_LOG_PATH
    min_score: float = Field(default_factory=default_min_score, ge=0.0, le=1.0)
    min_avg_score: float = Field(default_factory=default_min_avg_score, ge=0.0, le=1.0)
    min_distinct_docs: int = Field(default_factory=default_min_distinct_docs, ge=1, le=10)
    max_context_chars: int = Field(default_factory=default_max_context_chars, ge=200, le=12000)
    max_answer_chars: int = Field(default_factory=default_max_answer_chars, ge=120, le=4000)


class QueryResponse(BaseModel):
    query_id: str
    query: str
    query_type: str
    mode: str
    prompt_version: str
    decision: str
    answer: str
    clarifying_question: Optional[str] = None
    assessment: AssessmentOut
    citations: List[CitationOut]
    top_chunks: List[ChunkOut]
    tool_calls: List[ToolCallOut]
    workflow_steps: List[WorkflowStepOut]
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
    title="Keraforge",
    version="0.6.0",
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
  <title>Keraforge</title>
  <style>
    :root {
      --bg: #f5eadb;
      --bg-deep: #e7d5bf;
      --panel: rgba(255, 249, 241, 0.76);
      --panel-strong: rgba(255, 251, 246, 0.92);
      --ink: #1d1711;
      --muted: #665c54;
      --line: rgba(48, 35, 22, 0.12);
      --accent: #bf4f2f;
      --accent-strong: #92381f;
      --teal: #1f5f62;
      --olive: #66753a;
      --danger: #9b2c2c;
      --shadow: 0 28px 90px rgba(63, 42, 21, 0.16);
      --radius-lg: 28px;
      --radius-md: 20px;
      --radius-sm: 14px;
      --mono: "SFMono-Regular", "Menlo", monospace;
      --sans: "Avenir Next", "Segoe UI", sans-serif;
      --serif: "Iowan Old Style", "Palatino Linotype", serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 0%, rgba(191, 79, 47, 0.22), transparent 24%),
        radial-gradient(circle at 88% 14%, rgba(31, 95, 98, 0.20), transparent 24%),
        linear-gradient(135deg, var(--bg), var(--bg-deep));
      padding: 24px;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px);
      background-size: 28px 28px;
      mask-image: linear-gradient(to bottom, rgba(0,0,0,0.55), transparent 70%);
    }
    .shell {
      max-width: 1320px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
      position: relative;
      z-index: 1;
    }
    .hero,
    .panel {
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      backdrop-filter: blur(16px);
    }
    .hero {
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(0, 1.3fr) minmax(260px, 0.7fr);
      padding: 28px;
      border-radius: var(--radius-lg);
      background:
        linear-gradient(135deg, rgba(255, 249, 243, 0.95), rgba(243, 228, 209, 0.9));
      overflow: hidden;
      position: relative;
    }
    .hero::after {
      content: "";
      position: absolute;
      right: -64px;
      bottom: -76px;
      width: 260px;
      height: 260px;
      border-radius: 36px;
      background: linear-gradient(135deg, rgba(31, 95, 98, 0.18), rgba(191, 79, 47, 0.06));
      transform: rotate(18deg);
    }
    .hero-copy {
      display: grid;
      gap: 14px;
      align-content: start;
    }
    .eyebrow {
      font-size: 11px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--teal);
      font-weight: 700;
    }
    h1 {
      margin: 0;
      font-family: var(--serif);
      font-size: clamp(2.4rem, 6vw, 5.2rem);
      line-height: 0.95;
      max-width: 11ch;
    }
    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 62ch;
      font-size: 15px;
      line-height: 1.55;
    }
    .hero-stats {
      display: grid;
      gap: 12px;
      align-content: start;
      position: relative;
      z-index: 1;
    }
    .stat-card {
      background: rgba(255, 255, 255, 0.68);
      border: 1px solid rgba(48, 35, 22, 0.1);
      border-radius: var(--radius-md);
      padding: 16px 18px;
    }
    .stat-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
      margin-bottom: 6px;
    }
    .stat-value {
      font-size: 24px;
      font-weight: 700;
    }
    .stat-value small {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      margin-top: 6px;
    }
    .layout {
      display: grid;
      gap: 18px;
      grid-template-columns: minmax(320px, 430px) minmax(0, 1fr);
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border-radius: var(--radius-lg);
      padding: 22px;
    }
    .panel-title {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
    }
    .panel-title h2,
    .panel-title h3 {
      margin: 0;
      font-size: 18px;
      font-weight: 700;
    }
    .panel-kicker {
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    form {
      display: grid;
      gap: 16px;
    }
    label {
      display: grid;
      gap: 8px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 700;
    }
    input,
    select,
    textarea,
    button {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: var(--radius-sm);
      padding: 13px 14px;
      font: inherit;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.8);
      transition: border-color 150ms ease, transform 150ms ease, box-shadow 150ms ease;
    }
    textarea {
      min-height: 120px;
      resize: vertical;
      line-height: 1.45;
    }
    input:focus,
    select:focus,
    textarea:focus {
      outline: none;
      border-color: rgba(31, 95, 98, 0.4);
      box-shadow: 0 0 0 4px rgba(31, 95, 98, 0.10);
    }
    .grid-2,
    .grid-3 {
      display: grid;
      gap: 12px;
    }
    .grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .pill-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .preset {
      border: 1px solid rgba(31, 95, 98, 0.18);
      background: rgba(255, 255, 255, 0.72);
      color: var(--teal);
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }
    .preset:hover,
    button:hover {
      transform: translateY(-1px);
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .primary {
      cursor: pointer;
      border: none;
      color: white;
      background: linear-gradient(135deg, var(--accent), #df7d3d);
      font-weight: 800;
      letter-spacing: 0.02em;
    }
    .secondary {
      cursor: pointer;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.72);
      font-weight: 700;
    }
    .status-note {
      font-size: 13px;
      color: var(--muted);
      min-height: 18px;
    }
    .result-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }
    .status-cluster {
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 9px 14px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 800;
      border: 1px solid transparent;
    }
    .badge.grounded {
      background: rgba(102, 117, 58, 0.14);
      color: var(--olive);
      border-color: rgba(102, 117, 58, 0.18);
    }
    .badge.abstain {
      background: rgba(191, 79, 47, 0.12);
      color: var(--accent-strong);
      border-color: rgba(191, 79, 47, 0.18);
    }
    .badge.error {
      background: rgba(155, 44, 44, 0.12);
      color: var(--danger);
      border-color: rgba(155, 44, 44, 0.18);
    }
    .meta-line {
      display: flex;
      gap: 10px 16px;
      flex-wrap: wrap;
      font-size: 13px;
      color: var(--muted);
    }
    .answer {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 22px;
      min-height: 160px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
    }
    .answer h3 {
      margin: 0 0 10px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
    }
    .answer-body {
      font-size: 15px;
      line-height: 1.65;
      white-space: pre-wrap;
    }
    .clarifier {
      margin-top: 16px;
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(31, 95, 98, 0.08);
      border: 1px solid rgba(31, 95, 98, 0.12);
      font-size: 14px;
      line-height: 1.5;
    }
    .section-grid {
      display: grid;
      gap: 18px;
      margin-top: 18px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .subpanel {
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 18px;
    }
    .subpanel h3 {
      margin: 0 0 12px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .metric-grid {
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .metric {
      background: rgba(255,255,255,0.72);
      border-radius: 16px;
      padding: 12px;
      border: 1px solid rgba(48, 35, 22, 0.08);
    }
    .metric .key {
      display: block;
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }
    .metric .value {
      font-size: 22px;
      font-weight: 800;
    }
    .stack {
      display: grid;
      gap: 10px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.78);
    }
    .card-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
      margin-bottom: 8px;
    }
    .card-title {
      font-weight: 800;
      font-size: 14px;
    }
    .card-meta {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.45;
    }
    .chip {
      display: inline-block;
      border-radius: 999px;
      padding: 5px 9px;
      background: rgba(31, 95, 98, 0.09);
      color: var(--teal);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .snippet {
      margin-top: 8px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.55;
      white-space: pre-wrap;
    }
    .timeline {
      display: grid;
      gap: 10px;
    }
    .timeline-item {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: start;
      background: rgba(255, 255, 255, 0.74);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px 14px;
    }
    .timeline-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--teal);
      margin-top: 5px;
    }
    .timeline-copy strong {
      display: block;
      font-size: 14px;
      margin-bottom: 3px;
    }
    .timeline-copy span {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      word-break: break-word;
    }
    .detail-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }
    .detail-pill {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      padding: 6px 9px;
      border-radius: 999px;
      background: rgba(31, 95, 98, 0.08);
      border: 1px solid rgba(31, 95, 98, 0.12);
      color: var(--teal);
      font-size: 11px;
      line-height: 1.2;
      max-width: 100%;
    }
    .detail-pill strong {
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 10px;
    }
    .timeline-time {
      color: var(--muted);
      font-size: 12px;
      font-family: var(--mono);
      white-space: nowrap;
    }
    .empty {
      padding: 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.54);
      border: 1px dashed rgba(48, 35, 22, 0.14);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.6;
      color: #2b241e;
    }
    .json-wrap {
      max-height: 480px;
      overflow: auto;
      border-radius: 18px;
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--line);
      padding: 14px;
    }
    .footer-note {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.5;
    }
    @media (max-width: 1120px) {
      .layout,
      .hero,
      .section-grid {
        grid-template-columns: 1fr;
      }
      h1 { max-width: 100%; }
    }
    @media (max-width: 720px) {
      body { padding: 14px; }
      .grid-2,
      .grid-3,
      .metric-grid {
        grid-template-columns: 1fr;
      }
      .timeline-item {
        grid-template-columns: auto 1fr;
      }
      .timeline-time {
        grid-column: 2;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">Practical GenAI Exposure</div>
        <h1>Keraforge</h1>
        <p>
          Keraforge is a local multilingual RAG workspace that demonstrates practical exposure to LLMs,
          embeddings, vector databases, and orchestration layers. Query the corpus, inspect grounded answers,
          and review the execution trace to show hands-on GenAI work instead of generic claims.
        </p>
      </div>
      <div class="hero-stats">
        <div class="stat-card">
          <div class="stat-label">Execution Mode</div>
          <div class="stat-value" id="hero-mode">workflow<small>Deterministic orchestration by default</small></div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Local Stack</div>
          <div class="stat-value" id="hero-health">checking<small>FastAPI, Qdrant, embeddings, and LLM path</small></div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Latest Run</div>
          <div class="stat-value" id="hero-latency">idle<small>Latency and runtime context appear here</small></div>
        </div>
      </div>
    </section>

    <div class="layout">
      <section class="panel">
        <div class="panel-title">
          <div>
            <div class="panel-kicker">Controls</div>
            <h2>Keraforge Console</h2>
          </div>
        </div>
        <form id="query-form">
          <label>
            Query
            <textarea id="query" name="query">politica de date sintetice</textarea>
          </label>

          <div class="pill-row">
            <button type="button" class="preset" data-query="politica de date sintetice" data-lang="RO">RO policy</button>
            <button type="button" class="preset" data-query="zgodność dane syntetyczne" data-lang="PL">PL policy</button>
            <button type="button" class="preset" data-query="compliance synthetic datasets" data-lang="EN">EN compliance</button>
          </div>

          <div class="grid-2">
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
              Doc Type
              <input id="doc_type" name="doc_type" type="text" placeholder="policy">
            </label>
          </div>

          <div class="grid-3">
            <label>
              LLM
              <select id="llm" name="llm">
                <option value="ollama" selected>ollama</option>
                <option value="openai">openai</option>
              </select>
            </label>
            <label>
              Mode
              <select id="mode" name="mode">
                <option value="workflow" selected>workflow</option>
                <option value="agent">agent</option>
              </select>
            </label>
            <label>
              Device
              <select id="device" name="device">
                <option value="auto" selected>auto</option>
                <option value="mps">mps</option>
                <option value="cpu">cpu</option>
                <option value="cuda">cuda</option>
              </select>
            </label>
          </div>

          <div class="grid-3">
            <label>
              Top K
              <input id="top_k" name="top_k" type="number" min="1" max="10" value="3">
            </label>
            <label>
              Max Tool Calls
              <input id="max_tool_calls" name="max_tool_calls" type="number" min="1" max="4" value="2">
            </label>
            <label>
              Min Distinct Docs
              <input id="min_distinct_docs" name="min_distinct_docs" type="number" min="1" max="10" value="1">
            </label>
          </div>

          <div class="grid-2">
            <label>
              Min Score
              <input id="min_score" name="min_score" type="number" min="0" max="1" step="0.01" value="0.35">
            </label>
            <label>
              Min Avg Score
              <input id="min_avg_score" name="min_avg_score" type="number" min="0" max="1" step="0.01" value="0.25">
            </label>
          </div>

          <div class="grid-2">
            <label>
              Max Context Chars
              <input id="max_context_chars" name="max_context_chars" type="number" min="200" max="12000" step="100" value="900">
            </label>
            <label>
              Max Answer Chars
              <input id="max_answer_chars" name="max_answer_chars" type="number" min="120" max="4000" step="50" value="900">
            </label>
          </div>

          <div class="actions">
            <button type="submit" class="primary">Run Keraforge Query</button>
            <button type="button" class="secondary" id="copy-json">Copy Raw JSON</button>
          </div>
          <div class="status-note" id="status-note">Idle. Run a query to inspect how Keraforge retrieves, grounds, and answers.</div>
        </form>
      </section>

      <section class="panel">
        <div class="result-header">
          <div>
            <div class="panel-kicker">Result</div>
            <h2 id="result-title">Keraforge is ready</h2>
          </div>
          <div class="status-cluster">
            <span class="badge abstain" id="decision-badge">waiting</span>
            <div class="meta-line" id="result-meta">
              <span>query_id: pending</span>
            </div>
          </div>
        </div>

        <div class="answer">
          <h3>Answer</h3>
          <div class="answer-body" id="answer-body">Run a query to render the grounded answer, retrieval posture, and trace details here.</div>
          <div class="clarifier" id="clarifier" hidden></div>
        </div>

        <div class="section-grid">
          <div class="subpanel">
            <h3>Assessment</h3>
            <div class="metric-grid" id="assessment-grid">
              <div class="metric"><span class="key">Grounded</span><span class="value">-</span></div>
              <div class="metric"><span class="key">Top score</span><span class="value">-</span></div>
              <div class="metric"><span class="key">Avg score</span><span class="value">-</span></div>
              <div class="metric"><span class="key">Distinct docs</span><span class="value">-</span></div>
            </div>
            <div class="stack" id="assessment-reasons" style="margin-top:12px;">
              <div class="empty">Guardrail reasons appear here.</div>
            </div>
          </div>

          <div class="subpanel">
            <h3>Execution Trace</h3>
            <div class="timeline" id="workflow-list">
              <div class="empty">Execution steps, statuses, and timings appear here.</div>
            </div>
          </div>

          <div class="subpanel">
            <h3>Citations</h3>
            <div class="stack" id="citations-list">
              <div class="empty">Citations will render here once retrieval succeeds.</div>
            </div>
          </div>

          <div class="subpanel">
            <h3>Tool Calls</h3>
            <div class="stack" id="tool-calls-list">
              <div class="empty">Agent and workflow tool usage appears here.</div>
            </div>
          </div>

          <div class="subpanel" style="grid-column: 1 / -1;">
            <h3>Retrieved Chunks</h3>
            <div class="stack" id="chunks-list">
              <div class="empty">Top chunks and snippets appear here.</div>
            </div>
          </div>

          <div class="subpanel" style="grid-column: 1 / -1;">
            <h3>Raw JSON</h3>
            <div class="json-wrap">
              <pre id="raw-output">Submit a query to inspect the API payload.</pre>
            </div>
          </div>
        </div>

        <p class="footer-note">
          Keraforge uses the same guarded retrieval and generation path as the CLI and API. Weak retrieval should abstain instead of fabricating confidence.
        </p>
      </section>
    </div>
  </div>

  <script>
    const form = document.getElementById("query-form");
    const rawOutput = document.getElementById("raw-output");
    const statusNote = document.getElementById("status-note");
    const heroMode = document.getElementById("hero-mode");
    const heroHealth = document.getElementById("hero-health");
    const heroLatency = document.getElementById("hero-latency");
    const decisionBadge = document.getElementById("decision-badge");
    const resultTitle = document.getElementById("result-title");
    const resultMeta = document.getElementById("result-meta");
    const answerBody = document.getElementById("answer-body");
    const clarifier = document.getElementById("clarifier");
    const assessmentGrid = document.getElementById("assessment-grid");
    const assessmentReasons = document.getElementById("assessment-reasons");
    const workflowList = document.getElementById("workflow-list");
    const citationsList = document.getElementById("citations-list");
    const toolCallsList = document.getElementById("tool-calls-list");
    const chunksList = document.getElementById("chunks-list");
    const copyJsonButton = document.getElementById("copy-json");

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function fmt(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "-";
      }
      return Number(value).toFixed(digits);
    }

    function renderEmpty(container, message) {
      container.innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
    }

    function readValue(id) {
      return document.getElementById(id).value.trim();
    }

    function readNumber(id) {
      const value = document.getElementById(id).value;
      return value === "" ? null : Number(value);
    }

    function setDecision(decision) {
      decisionBadge.textContent = decision || "waiting";
      decisionBadge.className = "badge " + (
        decision === "grounded" ? "grounded" :
        decision === "abstain" ? "abstain" : "error"
      );
    }

    function renderAssessment(assessment) {
      if (!assessment) {
        assessmentGrid.innerHTML = '<div class="metric"><span class="key">Grounded</span><span class="value">-</span></div>';
        renderEmpty(assessmentReasons, "Guardrail reasons appear here.");
        return;
      }

      assessmentGrid.innerHTML = `
        <div class="metric"><span class="key">Grounded</span><span class="value">${assessment.grounded ? "yes" : "no"}</span></div>
        <div class="metric"><span class="key">Top score</span><span class="value">${fmt(assessment.top_score)}</span></div>
        <div class="metric"><span class="key">Avg score</span><span class="value">${fmt(assessment.avg_score)}</span></div>
        <div class="metric"><span class="key">Distinct docs</span><span class="value">${escapeHtml(assessment.distinct_docs)}</span></div>
      `;

      if (!assessment.reasons || assessment.reasons.length === 0) {
        renderEmpty(assessmentReasons, "No guardrail reasons were returned.");
        return;
      }

      assessmentReasons.innerHTML = assessment.reasons.map((reason) => `
        <div class="card">
          <div class="card-meta">${escapeHtml(reason)}</div>
        </div>
      `).join("");
    }

    function renderWorkflow(steps) {
      if (!steps || steps.length === 0) {
        renderEmpty(workflowList, "Execution steps, statuses, and timings appear here.");
        return;
      }

      const formatStepValue = (value) => {
        if (value === null || value === undefined || value === "") {
          return "-";
        }
        if (Array.isArray(value)) {
          return value.length === 0 ? "-" : value.join(", ");
        }
        if (typeof value === "object") {
          const keys = Object.keys(value);
          return keys.length === 0 ? "-" : keys.slice(0, 3).join(", ");
        }
        const text = String(value);
        return text.length > 48 ? text.slice(0, 45) + "..." : text;
      };

      const formatStepDetails = (details) => {
        const entries = Object.entries(details || {});
        if (entries.length === 0) {
          return '<span>No additional details.</span>';
        }

        return `<div class="detail-list">${entries.map(([key, value]) => `
          <span class="detail-pill"><strong>${escapeHtml(key)}</strong>${escapeHtml(formatStepValue(value))}</span>
        `).join("")}</div>`;
      };

      workflowList.innerHTML = steps.map((step) => `
        <div class="timeline-item">
          <div class="timeline-dot"></div>
          <div class="timeline-copy">
            <strong>${escapeHtml(step.step)}</strong>
            <span>${escapeHtml(step.status || "done")}</span>
            ${formatStepDetails(step.details || {})}
          </div>
          <div class="timeline-time">${fmt(step.duration_ms)} ms</div>
        </div>
      `).join("");
    }

    function renderCitations(citations) {
      if (!citations || citations.length === 0) {
        renderEmpty(citationsList, "Citations will render here once retrieval succeeds.");
        return;
      }

      citationsList.innerHTML = citations.map((item) => `
        <div class="card">
          <div class="card-head">
            <div>
              <div class="card-title">${escapeHtml(item.title || item.path || item.src)}</div>
              <div class="card-meta">${escapeHtml(item.src)}</div>
            </div>
            <span class="chip">score ${fmt(item.score)}</span>
          </div>
          <div class="card-meta">lang ${escapeHtml(item.lang || "-")} · type ${escapeHtml(item.doc_type || "-")} · chunk ${escapeHtml(item.chunk_index ?? "-")}</div>
        </div>
      `).join("");
    }

    function renderToolCalls(toolCalls) {
      if (!toolCalls || toolCalls.length === 0) {
        renderEmpty(toolCallsList, "Agent and workflow tool usage appears here.");
        return;
      }

      toolCallsList.innerHTML = toolCalls.map((call, index) => `
        <div class="card">
          <div class="card-head">
            <div>
              <div class="card-title">${index + 1}. ${escapeHtml(call.tool)}</div>
              <div class="card-meta">${escapeHtml(call.status || "done")} · ${fmt(call.duration_ms)} ms</div>
            </div>
            <span class="chip">${escapeHtml(call.status || "ok")}</span>
          </div>
          <div class="card-meta">inputs: ${escapeHtml(JSON.stringify(call.inputs || {}))}</div>
          <div class="card-meta">outputs: ${escapeHtml(JSON.stringify(call.outputs || {}))}</div>
        </div>
      `).join("");
    }

    function renderChunks(chunks) {
      if (!chunks || chunks.length === 0) {
        renderEmpty(chunksList, "Top chunks and snippets appear here.");
        return;
      }

      chunksList.innerHTML = chunks.map((chunk) => `
        <div class="card">
          <div class="card-head">
            <div>
              <div class="card-title">${escapeHtml(chunk.title || chunk.src)}</div>
              <div class="card-meta">${escapeHtml(chunk.src)}</div>
            </div>
            <span class="chip">score ${fmt(chunk.score)}</span>
          </div>
          <div class="snippet">${escapeHtml(chunk.snippet || chunk.text || "")}</div>
        </div>
      `).join("");
    }

    function renderResult(data) {
      resultTitle.textContent = data.decision === "grounded" ? "Grounded answer returned" : "Guardrail response returned";
      heroMode.innerHTML = `${escapeHtml(data.mode || "-")}<small>${escapeHtml(data.query_type || "unknown")} query</small>`;
      heroLatency.innerHTML = `${fmt(data.latency_ms, 0)} ms<small>${escapeHtml(data.collection || "-")} · ${escapeHtml(data.device || "-")}</small>`;
      setDecision(data.decision);

      resultMeta.innerHTML = `
        <span>query_id: ${escapeHtml(data.query_id || "-")}</span>
        <span>prompt: ${escapeHtml(data.prompt_version || "-")}</span>
        <span>llm: ${escapeHtml(data.llm || "-")}</span>
        <span>context: ${escapeHtml(data.context_chars ?? "-")} chars</span>
      `;

      answerBody.textContent = data.answer || "No answer returned.";
      if (data.clarifying_question) {
        clarifier.hidden = false;
        clarifier.textContent = "Clarifying question: " + data.clarifying_question;
      } else {
        clarifier.hidden = true;
        clarifier.textContent = "";
      }

      renderAssessment(data.assessment);
      renderWorkflow(data.workflow_steps);
      renderCitations(data.citations);
      renderToolCalls(data.tool_calls);
      renderChunks(data.top_chunks);
      rawOutput.textContent = JSON.stringify(data, null, 2);
    }

    function renderError(status, detail) {
      const message = detail || "Request failed.";
      setDecision("error");
      resultTitle.textContent = "Request failed";
      answerBody.textContent = message;
      clarifier.hidden = true;
      resultMeta.innerHTML = `<span>status: ${escapeHtml(status)}</span>`;
      heroLatency.innerHTML = `failed<small>inspect error details below</small>`;
      rawOutput.textContent = JSON.stringify({ status, detail }, null, 2);
      renderAssessment(null);
      renderEmpty(workflowList, "No workflow trace was returned.");
      renderEmpty(citationsList, "No citations were returned.");
      renderEmpty(toolCallsList, "No tool-call details were returned.");
      renderEmpty(chunksList, "No retrieved chunks were returned.");
    }

    async function checkHealth() {
      try {
        const response = await fetch("/health");
        const data = await response.json();
        heroHealth.innerHTML = `${escapeHtml(data.status || "ok")}<small>FastAPI route is reachable</small>`;
      } catch (error) {
        heroHealth.innerHTML = `offline<small>${escapeHtml(error.message)}</small>`;
      }
    }

    function buildPayload() {
      return {
        query: readValue("query"),
        lang: readValue("lang") || null,
        doc_type: readValue("doc_type") || null,
        llm: readValue("llm"),
        device: readValue("device"),
        mode: readValue("mode"),
        top_k: readNumber("top_k"),
        max_tool_calls: readNumber("max_tool_calls"),
        min_distinct_docs: readNumber("min_distinct_docs"),
        min_score: readNumber("min_score"),
        min_avg_score: readNumber("min_avg_score"),
        max_context_chars: readNumber("max_context_chars"),
        max_answer_chars: readNumber("max_answer_chars")
      };
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const payload = buildPayload();
      statusNote.textContent = "Running Keraforge against POST /query...";
      setDecision("abstain");
      resultTitle.textContent = "Waiting for response";
      answerBody.textContent = "Running query...";
      clarifier.hidden = true;
      rawOutput.textContent = JSON.stringify(payload, null, 2);

      try {
        const response = await fetch("/query", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        const data = await response.json();
        if (!response.ok) {
          statusNote.textContent = "Request failed.";
          renderError(response.status, data.detail || "Unknown error");
          return;
        }

        statusNote.textContent = "Completed successfully.";
        renderResult(data);
      } catch (error) {
        statusNote.textContent = "Network or browser error.";
        renderError("network", error.message);
      }
    });

    document.querySelectorAll(".preset").forEach((button) => {
      button.addEventListener("click", () => {
        document.getElementById("query").value = button.dataset.query || "";
        document.getElementById("lang").value = button.dataset.lang || "";
      });
    });

    document.getElementById("mode").addEventListener("change", (event) => {
      heroMode.innerHTML = `${escapeHtml(event.target.value)}<small>Pending next query</small>`;
    });

    copyJsonButton.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(rawOutput.textContent);
        statusNote.textContent = "Raw JSON copied to clipboard.";
      } catch (error) {
        statusNote.textContent = "Clipboard write failed in this browser.";
      }
    });

    checkHealth();
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

    result["latency_ms"] = round((perf_counter() - started) * 1000, 2)
    return QueryResponse(**result)
