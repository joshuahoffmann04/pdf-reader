"""MARley frontend: a single-file web app for chatting with the full pipeline."""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if load_dotenv:
    load_dotenv(ROOT / ".env")

from chunking import ChunkingResult  # noqa: E402
from generation.citations import normalize_and_select_citations  # noqa: E402
from generation.context_builder import build_context  # noqa: E402
from generation.json_utils import safe_parse_json  # noqa: E402
from generation.ollama_client import chat  # noqa: E402
from generation.postprocess import postprocess_answer  # noqa: E402
from generation.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402
from retrieval.bm25_index import BM25Index  # noqa: E402
from retrieval.embedder import OllamaEmbedder  # noqa: E402
from retrieval.hybrid import rrf_merge  # noqa: E402
from retrieval.models import RetrievalHit  # noqa: E402
from retrieval.rerank import rerank_hits  # noqa: E402
from retrieval.vector_index import VectorIndex  # noqa: E402

import chromadb  # noqa: E402


@dataclass
class MarleyConfig:
    host: str = os.environ.get("MARLEY_HOST", "127.0.0.1")
    port: int = int(os.environ.get("MARLEY_PORT", "8090"))
    chunks_path: str = os.environ.get("MARLEY_CHUNKS_PATH", "")
    retrieval_mode: str = os.environ.get("MARLEY_RETRIEVAL_MODE", "hybrid")
    top_k: int = int(os.environ.get("MARLEY_TOP_K", "8"))
    max_context_tokens: int = int(os.environ.get("MARLEY_MAX_CONTEXT_TOKENS", "2048"))
    output_tokens: int = int(os.environ.get("MARLEY_OUTPUT_TOKENS", "512"))
    embed_model: str = os.environ.get("MARLEY_EMBED_MODEL", "nomic-embed-text")
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.1:latest")
    rrf_k: int = int(os.environ.get("MARLEY_RRF_K", "60"))


@dataclass
class MarleyState:
    config: MarleyConfig
    document_id: str
    chunk_path: Path
    chunk_count: int
    bm25: BM25Index
    vector: VectorIndex


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str | None = None
    top_k: int | None = Field(None, ge=1, le=50)
    max_context_tokens: int | None = Field(None, ge=256, le=8192)
    output_tokens: int | None = Field(None, ge=128, le=2048)


def _find_latest_chunk_file() -> Path:
    candidates: list[Path] = []
    test_path = ROOT / "test" / "chunking" / "output" / "chunks.json"
    if test_path.exists():
        candidates.append(test_path)

    data_root = ROOT / "data" / "chunking"
    if data_root.exists():
        candidates.extend(data_root.rglob("chunks/*.json"))

    if not candidates:
        raise FileNotFoundError(
            "No chunk file found. Set MARLEY_CHUNKS_PATH or generate chunks first."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_state() -> MarleyState:
    config = MarleyConfig()
    chunk_path = Path(config.chunks_path) if config.chunks_path else _find_latest_chunk_file()
    chunk_result = ChunkingResult.load(str(chunk_path))

    chunks: list[dict[str, Any]] = []
    for chunk in chunk_result.chunks:
        chunks.append(
            {
                "document_id": chunk_result.document_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata.model_dump(),
            }
        )

    bm25 = BM25Index()
    bm25.build(chunks)

    embedder = OllamaEmbedder(model=config.embed_model, base_url=config.ollama_base_url)
    chroma_client = chromadb.Client()
    vector = VectorIndex(
        persist_directory=":memory:",
        collection_name="marley_chunks",
        embedder=embedder,
        chroma_client=chroma_client,
    )
    vector.ingest(chunks)

    return MarleyState(
        config=config,
        document_id=chunk_result.document_id,
        chunk_path=chunk_path,
        chunk_count=len(chunks),
        bm25=bm25,
        vector=vector,
    )


STATE = _load_state()
APP = FastAPI(title="MARley Chatbot")


def _retrieve_hits(query: str, mode: str, top_k: int) -> list[RetrievalHit]:
    filters = {"document_id": STATE.document_id}
    if mode == "bm25":
        return STATE.bm25.search(query, top_k=top_k, filters=filters)
    if mode == "vector":
        return STATE.vector.search(query, top_k=top_k, filters=filters)
    if mode == "hybrid":
        # Pull a larger candidate set, merge via RRF, then rerank deterministically.
        # This improves recall for answer-bearing chunks that might otherwise be rank 10-30.
        candidate_k = min(max(top_k * 6, 50), 120)
        bm25_hits = STATE.bm25.search(query, top_k=candidate_k, filters=filters)
        vector_hits = STATE.vector.search(query, top_k=candidate_k, filters=filters)
        merge_k = min(candidate_k * 2, len(bm25_hits) + len(vector_hits))
        merged = rrf_merge(bm25_hits, vector_hits, merge_k, STATE.config.rrf_k)
        return rerank_hits(query, merged)[:top_k]
    raise ValueError(f"Unsupported mode: {mode}")


@APP.get("/", response_class=HTMLResponse)
def index() -> str:
    return _HTML


@APP.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "document_id": STATE.document_id,
        "chunk_count": STATE.chunk_count,
        "chunk_path": str(STATE.chunk_path),
        "default_mode": STATE.config.retrieval_mode,
        "top_k": STATE.config.top_k,
        "max_context_tokens": STATE.config.max_context_tokens,
        "ollama_model": STATE.config.ollama_model,
        "embed_model": STATE.config.embed_model,
    }


@APP.post("/api/chat")
def chat_api(request: ChatRequest) -> dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    mode = (request.mode or STATE.config.retrieval_mode).strip().lower()
    top_k = request.top_k or STATE.config.top_k
    max_context_tokens = request.max_context_tokens or STATE.config.max_context_tokens
    output_tokens = request.output_tokens or STATE.config.output_tokens

    hits = _retrieve_hits(query, mode, top_k)
    results = [hit.model_dump() for hit in hits]

    context = build_context(
        query=query,
        results=results,
        max_context_tokens=max_context_tokens,
        system_prompt=SYSTEM_PROMPT,
        user_template=USER_PROMPT_TEMPLATE,
    )

    user_prompt = USER_PROMPT_TEMPLATE.format(query=query, context=context.context_text)
    content = chat(
        base_url=STATE.config.ollama_base_url,
        model=STATE.config.ollama_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_schema=RESPONSE_SCHEMA,
        temperature=0.2,
        output_tokens=output_tokens,
    )

    payload = safe_parse_json(content)

    answer = postprocess_answer((payload.get("answer") or "").strip(), query)
    missing_info = (payload.get("missing_info") or "").strip()
    citations = normalize_and_select_citations(
        payload.get("citations", []),
        context.selected_chunks,
        answer=answer,
        query=query,
    )
    if answer:
        missing_info = ""
    elif not missing_info:
        missing_info = "Information nicht im Dokument enthalten."

    return {
        "answer": answer,
        "missing_info": missing_info,
        "citations": citations,
        "metadata": {
            "mode": mode,
            "top_k": top_k,
            "selected_chunks": len(context.selected_chunks),
            "used_tokens": context.used_tokens,
            "available_tokens": context.available_tokens,
        },
    }


def _open_browser(host: str, port: int) -> None:
    time.sleep(0.6)
    webbrowser.open(f"http://{host}:{port}")


def run() -> None:
    import uvicorn

    threading.Thread(
        target=_open_browser,
        args=(STATE.config.host, STATE.config.port),
        daemon=True,
    ).start()

    uvicorn.run(APP, host=STATE.config.host, port=STATE.config.port, log_level="info")


_HTML = """
<!doctype html>
<html lang="de">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>MARley Chatbot</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600&family=Inter:wght@400;600&display=swap" rel="stylesheet" />
    <style>
      :root {
        --ink: #0f172a;
        --muted: #475569;
        --surface: rgba(255, 255, 255, 0.9);
        --accent: #1d4ed8;
        --accent-soft: rgba(29, 78, 216, 0.1);
        --line: rgba(148, 163, 184, 0.35);
        --shadow: 0 18px 50px rgba(15, 23, 42, 0.15);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Inter", "Segoe UI", sans-serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top, rgba(226, 232, 240, 0.9), transparent 60%),
          linear-gradient(120deg, #f8fafc, #e2e8f0);
        min-height: 100vh;
      }
      header {
        padding: 28px 40px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      h1 {
        font-family: "Fraunces", serif;
        font-weight: 600;
        font-size: 32px;
        margin: 0;
      }
      .tagline {
        color: var(--muted);
        margin-top: 4px;
        font-size: 14px;
      }
      .status {
        font-size: 12px;
        color: var(--muted);
        padding: 6px 12px;
        border-radius: 999px;
        background: var(--surface);
        border: 1px solid var(--line);
      }
      main {
        display: grid;
        grid-template-columns: 2.2fr 1fr;
        gap: 24px;
        padding: 0 40px 40px;
      }
      .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 20px;
        box-shadow: var(--shadow);
        padding: 24px;
        backdrop-filter: blur(10px);
      }
      .chat-window {
        display: flex;
        flex-direction: column;
        height: 70vh;
      }
      .messages {
        flex: 1;
        overflow-y: auto;
        padding-right: 6px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }
      .bubble {
        padding: 14px 16px;
        border-radius: 16px;
        max-width: 85%;
        line-height: 1.5;
        font-size: 14px;
      }
      .user {
        align-self: flex-end;
        background: var(--accent);
        color: white;
        border-bottom-right-radius: 4px;
      }
      .assistant {
        align-self: flex-start;
        background: var(--accent-soft);
        color: var(--ink);
        border-bottom-left-radius: 4px;
      }
      form {
        display: flex;
        gap: 12px;
        margin-top: 18px;
      }
      textarea {
        flex: 1;
        resize: none;
        border-radius: 14px;
        border: 1px solid var(--line);
        padding: 12px 14px;
        font-family: inherit;
        font-size: 14px;
        background: white;
      }
      button {
        border: none;
        background: var(--accent);
        color: white;
        padding: 12px 18px;
        border-radius: 14px;
        font-weight: 600;
        cursor: pointer;
      }
      button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      .section-title {
        font-weight: 600;
        margin-bottom: 12px;
        font-size: 14px;
      }
      .kv {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
        margin-bottom: 8px;
        color: var(--muted);
      }
      select, input[type="number"] {
        width: 100%;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--line);
        margin-bottom: 14px;
      }
      .citations {
        display: flex;
        flex-direction: column;
        gap: 12px;
        max-height: 45vh;
        overflow-y: auto;
      }
      .citation {
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px 12px;
        background: white;
        font-size: 12px;
        color: var(--muted);
      }
      .citation strong {
        color: var(--ink);
      }
      pre {
        background: #0f172a;
        color: #e2e8f0;
        padding: 12px;
        border-radius: 12px;
        font-size: 11px;
        overflow-x: auto;
        white-space: pre-wrap;
      }
      @media (max-width: 980px) {
        main {
          grid-template-columns: 1fr;
        }
        header {
          flex-direction: column;
          align-items: flex-start;
          gap: 12px;
        }
      }
    </style>
  </head>
  <body>
    <header>
      <div>
        <h1>MARley</h1>
        <div class="tagline">Dein lokaler Dokument-Chatbot</div>
      </div>
      <div class="status" id="status">Lade Kontext ...</div>
    </header>
    <main>
      <section class="panel chat-window">
        <div class="messages" id="messages"></div>
        <form id="chat-form">
          <textarea id="prompt" rows="3" placeholder="Stell eine Frage zum Dokument ..."></textarea>
          <button id="send-btn" type="submit">Senden</button>
        </form>
      </section>
      <aside class="panel">
        <div class="section-title">Einstellungen</div>
        <label>Retrieval Mode</label>
        <select id="mode">
          <option value="hybrid">hybrid</option>
          <option value="bm25">bm25</option>
          <option value="vector">vector</option>
        </select>
        <label>Top-K</label>
        <input id="top_k" type="number" min="1" max="50" value="8" />
        <label>Max Context Tokens</label>
        <input id="max_tokens" type="number" min="256" max="8192" value="2048" />
        <label>Output Tokens</label>
        <input id="out_tokens" type="number" min="128" max="2048" value="512" />

        <div class="section-title">Citations</div>
        <div class="citations" id="citations"></div>

        <div class="section-title">Antwort-JSON</div>
        <pre id="raw-json">{}</pre>
      </aside>
    </main>

    <script>
      const messages = document.getElementById("messages");
      const form = document.getElementById("chat-form");
      const prompt = document.getElementById("prompt");
      const sendBtn = document.getElementById("send-btn");
      const statusEl = document.getElementById("status");
      const citationsEl = document.getElementById("citations");
      const rawJsonEl = document.getElementById("raw-json");
      const modeEl = document.getElementById("mode");
      const topKEl = document.getElementById("top_k");
      const maxTokensEl = document.getElementById("max_tokens");
      const outTokensEl = document.getElementById("out_tokens");

      function addMessage(role, text) {
        const bubble = document.createElement("div");
        bubble.className = `bubble ${role}`;
        bubble.textContent = text;
        messages.appendChild(bubble);
        messages.scrollTop = messages.scrollHeight;
      }

      function renderCitations(citations) {
        citationsEl.innerHTML = "";
        if (!citations || citations.length === 0) {
          citationsEl.innerHTML = "<div class='kv'>Keine Zitate</div>";
          return;
        }
        citations.forEach((cite) => {
          const item = document.createElement("div");
          item.className = "citation";
          const pages = (cite.page_numbers || []).join(", ") || "unknown";
          item.innerHTML = `<strong>${cite.chunk_id}</strong><br/>Seiten: ${pages}<br/>${cite.snippet}`;
          citationsEl.appendChild(item);
        });
      }

      async function loadStatus() {
        try {
          const res = await fetch("/api/status");
          const data = await res.json();
          statusEl.textContent = `Dokument: ${data.document_id} - Chunks: ${data.chunk_count}`;
          modeEl.value = data.default_mode || "hybrid";
          topKEl.value = data.top_k || 8;
          maxTokensEl.value = data.max_context_tokens || 2048;
        } catch (e) {
          statusEl.textContent = "Status nicht erreichbar";
        }
      }

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const query = prompt.value.trim();
        if (!query) return;
        addMessage("user", query);
        prompt.value = "";
        sendBtn.disabled = true;
        statusEl.textContent = "Denke nach ...";

        const payload = {
          query,
          mode: modeEl.value,
          top_k: Number(topKEl.value),
          max_context_tokens: Number(maxTokensEl.value),
          output_tokens: Number(outTokensEl.value)
        };

        try {
          const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          if (!res.ok) {
            throw new Error(data.detail || "Server error");
          }
          addMessage("assistant", data.answer || data.missing_info || "Keine Antwort.");
          renderCitations(data.citations || []);
          rawJsonEl.textContent = JSON.stringify(data, null, 2);
          statusEl.textContent = "Bereit";
        } catch (err) {
          addMessage("assistant", "Fehler: " + err.message);
          statusEl.textContent = "Fehler";
        } finally {
          sendBtn.disabled = false;
        }
      });

      loadStatus();
    </script>
  </body>
</html>
"""


if __name__ == "__main__":
    run()
