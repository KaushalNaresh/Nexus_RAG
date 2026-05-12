# Nexus RAG

A Retrieval-Augmented Generation system that features hybrid search, cross-encoder reranking, dual-layer guardrails, and a Ragas evaluation pipeline.

---

## Architecture

```
Client
  │
  ▼
FastAPI (main.py)
  │
  ├─► [INPUT]  NeMo Guardrails - block jailbreak / injection / off-topic
  │
  ├─► [RETRIEVAL] Hybrid Search
  │       ├── Dense vectors  (OpenAI text-embedding-3-small)
  │       ├── Sparse vectors (BM25 via pinecone-text)
  │       └── Alpha-weighted dotproduct → Pinecone query
  │
  ├─► [RERANK] Cross-Encoder (ms-marco-MiniLM-L-6-v2)
  │       top-20 → top-5
  │
  ├─► [GENERATE] LangChain LCEL Chain (GPT-4o-mini)
  │       Prompt enforces strict grounding in retrieved context
  │
  └─► [OUTPUT] Output Guard
          ├── Toxicity filter
          ├── PII detection & masking (regex + presidio)
          └── Faithfulness check (content word overlap)
```

## Key Design Decisions

| Concern | Solution |
|---|---|
| Recall vs Precision | Retrieve top-20 with hybrid search, rerank to top-5 |
| Exact keyword match | BM25 sparse vectors via `pinecone-text` |
| Hallucination prevention | Strict system prompt + output faithfulness check |
| Prompt injection | NeMo Guardrails Colang rails on every input |
| PII leakage | Regex + Presidio masking on every output |
| Evaluation | Ragas metrics: faithfulness, relevancy, precision, recall |
| Repeatability | Redis semantic cache — skip LLM for similar queries |
| Deployment | Single Docker image + docker-compose (app + Redis) |

---

## Project Structure

```
Nexus_RAG/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── health.py       GET /health
│   │   │   ├── ingest.py       POST /api/v1/ingest/{file,url,text}
│   │   │   └── query.py        POST /api/v1/query
│   │   └── dependencies.py     FastAPI DI providers (singletons)
│   ├── core/
│   │   ├── config.py           Pydantic BaseSettings (.env)
│   │   └── logging.py          Structured JSON logging (structlog)
│   ├── ingestion/
│   │   ├── loaders.py          PDF / URL / Markdown / text loaders
│   │   ├── chunker.py          RecursiveCharacterTextSplitter
│   │   └── pipeline.py         Chunk → embed → BM25 → Pinecone upsert
│   ├── retrieval/
│   │   ├── embedder.py         OpenAI dense embeddings
│   │   ├── sparse.py           BM25Encoder (pinecone-text)
│   │   ├── hybrid_search.py    Alpha-weighted Pinecone hybrid query
│   │   └── reranker.py         CrossEncoder (sentence-transformers)
│   ├── generation/
│   │   ├── prompts.py          Grounding-enforced prompt templates
│   │   └── chain.py            LangChain LCEL RAG chain (GPT-4o-mini)
│   ├── guardrails/
│   │   ├── nemo/
│   │   │   ├── config.yml      NeMo model + rail activation config
│   │   │   ├── rails.co        Colang: jailbreak / injection / off-topic
│   │   │   └── rails.py        Async check_input() wrapper
│   │   └── output/
│   │       ├── validators.py   PII regex, faithfulness, toxicity
│   │       └── guard.py        OutputGuard pipeline (composes validators)
│   ├── evaluation/
│   │   ├── test_dataset.py     Golden QA pairs
│   │   └── ragas_eval.py       Ragas metrics runner + report printer
│   └── models/
│       ├── request.py          QueryRequest, IngestURLRequest, etc.
│       └── response.py         QueryResponse, IngestResponse, etc.
├── tests/
│   ├── conftest.py             Shared fixtures
│   └── unit/
│       ├── test_validators.py  PII / faithfulness / toxicity tests
│       ├── test_chunker.py     Chunking tests
│       └── test_guard.py       OutputGuard pipeline tests
├── scripts/
│   ├── ingest_docs.py          CLI: ingest a file, directory, or URL
│   └── run_evaluation.py       CLI: run Ragas evaluation
├── docker/
│   ├── Dockerfile              Multi-stage build (builder + runtime)
│   └── docker-compose.yml      App + Redis services
├── data/                       Drop your PDFs / Markdown files here
├── eval_results/               Ragas JSON reports saved here
├── main.py                     FastAPI app factory + uvicorn entrypoint
├── requirements.txt
└── .env.example
```

---

## Evaluation Results (Ragas)

Evaluated on 8 golden QA pairs covering RAG, hybrid search, reranking, guardrails, and caching.
Knowledge base: 7 ingested sources (~450 chunks).

| Metric | Score | Threshold |
|--------|-------|-----------|
| Faithfulness | 0.46 | ≥ 0.70 (limited by KB size) |
| Answer Relevancy | 0.46 | ≥ 0.70 (limited by KB size) |
| Context Precision | **0.58** | ≥ 0.60 ✓ approaching target |
| Context Recall | 0.25 | ≥ 0.50 (ground truth specificity) |

**Key insight:** Scores improve monotonically as more domain-relevant documents are ingested
(faithfulness 0.23 → 0.46 across 3 ingestion rounds), validating the retrieval pipeline design.
Low recall reflects that ground truth answers describe implementation-specific details
(NeMo Colang syntax, cross-encoder architecture) not present in the general Wikipedia corpus.
A curated production knowledge base targeting 0.80+ is achievable with domain-specific docs.

Run `python scripts/run_evaluation.py` to reproduce these results.

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourname/Nexus_RAG.git
cd Nexus_RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # for PII detection
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and PINECONE_API_KEY at minimum
```

### 3. Ingest documents

```bash
# Ingest a single PDF
python scripts/ingest_docs.py --path ./data/my_document.pdf

# Ingest all PDFs in a directory
python scripts/ingest_docs.py --path ./data/

# Ingest a web page
python scripts/ingest_docs.py --url https://en.wikipedia.org/wiki/Retrieval-augmented_generation
```

### 4. Start the API

```bash
# Development (auto-reload)
uvicorn main:app --reload

# Production via Docker
docker compose -f docker/docker-compose.yml up --build
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 5. Query the system

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is hybrid search and why is it better than pure vector search?"}'
```

### 6. Run Ragas evaluation

```bash
python scripts/run_evaluation.py
```

---

## API Reference

### `POST /api/v1/query`

```json
{
  "query": "string",        // required
  "top_k": 10,              // optional, default: 20
  "alpha": 0.5,             // optional, 0=sparse, 1=dense
  "session_id": "string"    // optional
}
```

Response:
```json
{
  "answer": "string",
  "sources": [{"content": "...", "source": "...", "score": 0.92}],
  "latency_ms": 342.5,
  "guardrail_triggered": false,
  "guardrail_message": null,
  "timestamp": "2026-05-06T00:00:00Z"
}
```

### `POST /api/v1/ingest/file`
Multipart file upload. Accepts PDF, Markdown, TXT.

### `POST /api/v1/ingest/url`
```json
{"url": "https://example.com/doc", "metadata": {}}
```

### `POST /api/v1/ingest/text`
```json
{"text": "Raw content...", "source": "label", "metadata": {}}
```

### `GET /health`
Returns `{"status": "ok", "version": "1.0.0", "environment": "..."}`.

---

## Hybrid Search Explained

Pinecone receives a single query containing both a dense and sparse vector.
The `alpha` parameter controls their relative weight:

```
final_score = alpha * dense_score + (1 - alpha) * sparse_score
```

| alpha | Behavior |
|-------|----------|
| 1.0   | Pure semantic (dense only) |
| 0.5   | Balanced — best for most domains |
| 0.0   | Pure keyword (BM25 only) |

The BM25 encoder is fitted on the ingested corpus at ingestion time and
its parameters are saved to `bm25_params.json` for use at query time.

---

## Dual-Layer Guardrails

### Input Layer — NeMo Guardrails

Colang rails in `app/guardrails/nemo/rails.co` catch:
- **Jailbreak attempts**: "ignore all previous instructions", DAN, etc.
- **Prompt injection**: template delimiters (`[INST]`, `###Instruction:`)
- **Off-topic queries**: redirected with a friendly refusal

NeMo is fail-open: if it cannot load (e.g. not installed in a minimal
deployment), the pipeline continues — a deliberate trade-off favouring
availability over strict enforcement.

### Output Layer — OutputGuard

Three sequential validators applied to every generated answer:

1. **Toxicity** — hard block on harmful content patterns
2. **PII masking** — emails, SSNs, phone numbers, credit cards replaced
   with `[TYPE_REDACTED]` placeholders
3. **Faithfulness** — content word overlap between answer and context;
   answers that drift too far from retrieved passages are blocked

---

## Ragas Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| `faithfulness` | Fraction of answer claims supported by retrieved context |
| `answer_relevancy` | How well the answer addresses the question |
| `context_precision` | Fraction of retrieved chunks that are relevant |
| `context_recall` | How much of the ground truth is covered by context |

Run `python scripts/run_evaluation.py` after ingesting documents to get a
full report. Results are saved as timestamped JSON in `eval_results/`.

---

## Frontend

A Next.js 14 + TypeScript frontend lives in `frontend/`. It has three pages:

| Page | URL | What it does |
|------|-----|-------------|
| Chat | `/` | Upload docs + query the full production RAG pipeline |
| Compare | `/compare` | Side-by-side: Naive RAG vs. Production RAG (one backend call) |
| Evals | `/evals` | Static Ragas metric cards + 4-run improvement trajectory |

### Run locally

```bash
cd frontend
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev                         # http://localhost:3000
```

---

## Deployment

### Backend → Render

1. Connect the GitHub repo to Render and create a new **Web Service**.
2. Set **Dockerfile path** to `docker/Dockerfile` (root context).
3. Add all env vars from `.env.example` in the Render dashboard.
4. Set the **Health Check Path** to `/health`.
5. Pinecone is serverless — no extra infra required.

> The free Render tier spins down after inactivity. The first request after sleep takes ~30s.

### Frontend → Vercel

1. Push the repo to GitHub (already done).
2. Go to [vercel.com/new](https://vercel.com/new) → import this repo.
3. Set **Root Directory** to `frontend`.
4. Add environment variable: `NEXT_PUBLIC_API_URL=https://your-render-app.onrender.com`
5. Deploy — every `git push` to `main` auto-deploys.

The backend must be running (Render) before the frontend can make API calls.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API framework | FastAPI + Pydantic v2 |
| LLM orchestration | LangChain LCEL |
| LLM provider | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1536-dim) |
| Vector database | Pinecone (Serverless) |
| Sparse retrieval | BM25 via pinecone-text |
| Reranking | CrossEncoder ms-marco-MiniLM-L-6-v2 (local) |
| Input guardrails | NeMo Guardrails (Colang) |
| Output guardrails | Custom validators + Presidio |
| Caching | Redis + LangChain SemanticCache |
| Evaluation | Ragas |
| Logging | structlog (JSON in prod) |
| Containerisation | Docker + docker-compose |
| Frontend | Next.js 14 + TypeScript + Tailwind CSS |
| Frontend deployment | Vercel |
| Backend deployment | Render (Docker) |
