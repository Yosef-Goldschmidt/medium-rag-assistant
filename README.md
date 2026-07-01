# Medium Article RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions **only** from a corpus of ~7,600 English Medium articles. It retrieves relevant passages from a vector database and has an LLM answer strictly from that retrieved context — never from the model's own background knowledge.

**Live URL:** `<LIVE_URL>`
**Repository:** `<GITHUB_URL>`

---

## Architecture

The system has two phases:

```
PHASE A — INDEXING (offline, run once)
  CSV → clean → chunk (token-based, with overlap) → embed (batched) → Pinecone

PHASE B — ANSWERING (live, per request)
  question → embed → Pinecone search (top-k, score filter)
           → (if "list distinct": widen pool + dedup by article)
           → build prompt → gpt-5-mini → JSON answer
```

The LLM is constrained by a mandatory system prompt to answer only from the retrieved context, and to reply *"I don't know based on the provided Medium articles data."* when the context is insufficient.

### Models
| Role | Model | Notes |
|---|---|---|
| Embeddings | `text-embedding-3-small` | 1536 dimensions |
| Chat | `gpt-5-mini` | answers from retrieved context only |

Both are served via an OpenAI-compatible proxy (`base_url`), configured through environment variables.

### Vector database
Pinecone serverless index `medium-rag` — **1536 dimensions, cosine** metric (AWS `us-east-1`).

---

## Modules (`src/`)

| File | Responsibility |
|---|---|
| `config.py` | Single source of truth for all hyperparameters |
| `data.py` | Load CSV, parse authors/tags, clean text (strips photo-caption noise) |
| `chunking.py` | Token-based sliding-window chunking (via `tiktoken`) |
| `embeddings.py` | Batched embedding client |
| `vector_store.py` | Pinecone: create index, upsert, query |
| `retriever.py` | Question → relevant chunks; score filtering; dedup by article |
| `prompt.py` | Mandatory system prompt + user-prompt builder |
| `llm.py` | Chat completion client |
| `rag.py` | Orchestrator — ties retrieval + prompt + LLM into the API response |
| `indexer.py` | Full-corpus indexing pipeline (batched, with retry) |
| `experiment.py` | Hyperparameter comparison harness (see Report below) |

---

## Hyperparameters (report)

Current configuration (also served live at `/api/stats`):

| Parameter | Value | Assignment limit |
|---|---|---|
| `chunk_size` | 512 tokens | ≤ 1024 |
| `overlap_ratio` | 0.20 | ≤ 0.30 |
| `top_k` | 8 | ≤ 30 |
| `min_score` | 0.20 | — (relevance filter) |

### How these were chosen

We compared configurations on a 200-article subset, indexing each into a separate Pinecone **namespace** so they could be evaluated side by side without re-embedding the whole corpus. Retrieval quality was judged on questions spanning all four required query types.

**1. Chunk size (overlap held roughly constant).**

| chunk_size | chunks / 200 articles | context @ top_k=5 | behaviour |
|---|---|---|---|
| 256 | 1140 | ~1,280 tok | **Highest precision**, but "peaky" — one article floods the top-k (bad for distinct-listing) |
| **512** | **685** | ~2,560 tok | **Balanced** — good precision, natural article diversity, enough context per chunk for summaries |
| 1024 | 370 | ~5,120 tok | Most diverse, but weaker precision and bloated context |

Smaller chunks give sharper matches but too little context for the *summary* and *recommendation* query types, and flood results with one article. Larger chunks waste context and dilute precision. **512 is the balanced middle**, so it was chosen.

**2. Overlap (chunk_size fixed at 512).**

| overlap | chunks / 200 articles | effect |
|---|---|---|
| 0.10 | 623 | Rankings essentially identical to 0.20 |
| **0.20** | **685** | Chosen — balanced |
| 0.30 | 770 | Only added redundant same-article chunks + storage; no ranking gain |

Overlap had **minimal effect on which article was retrieved**; higher overlap only increased redundancy and chunk count. 0.20 was chosen as a safe middle value.

**3. Relevance filter (`min_score`) and `top_k`.**
Initially a cutoff of 0.35 was tried (experiment scores: strong matches 0.42–0.67, weak ~0.32–0.40). But on the **full corpus** this over-filtered legitimate broad-topic queries (e.g. "education"), because valid-but-broad matches and genuine non-matches overlap around 0.33–0.37 — there is no clean separating threshold. It was therefore lowered to **0.20** (drops only clearly-unrelated chunks) and `top_k` raised from 5 to **8** for better recall. The "I don't know" guardrail is enforced by the LLM reading the context, not by the score filter, so loosening the filter did not weaken refusals (verified: out-of-corpus questions still correctly refuse).

**4. Distinct-article listing.**
Raw top-k returns the closest *chunks*, not distinct *articles*. For "list N distinct articles" queries, the system widens the pool (`LIST_POOL = 30`), deduplicates by `article_id` (keeping each article's best chunk), and hands the LLM up to `LIST_CANDIDATES = 8` distinct articles so it can select the ones actually on-topic (rather than being forced to use only the top 3 by raw score).

---

## API

### `POST /api/prompt`
Request:
```json
{ "question": "Your natural language question here" }
```
Response:
```json
{
  "response": "Final natural-language answer from the model.",
  "context": [
    { "article_id": "1234", "title": "Sample title", "chunk": "retrieved chunk", "score": 0.1234 }
  ],
  "Augmented_prompt": {
    "System": "the system prompt used to query the chat model",
    "User": "the user prompt used to query the chat model"
  }
}
```

### `GET /api/stats`
Response:
```json
{ "chunk_size": 512, "overlap_ratio": 0.2, "top_k": 8 }
```

---

## Setup & running

### Environment variables
Create a `.env` (local) or set in the Vercel dashboard:
```
NBUECSE_API_KEY=<embedding/chat proxy key>
NBUECSE_BASE_URL=<proxy base url, e.g. https://.../v1>
PINECONE_API_KEY=<pinecone key>
```

### Install
```
uv sync            # or: pip install -r requirements.txt
```

### Build the index (Phase A, run once)
Download the dataset CSV, place it at the repo root, then:
```
python src/indexer.py
```
This streams every article, chunks at the configured settings, embeds in batches of 100, and upserts into Pinecone. It is idempotent (safe to re-run) and retries transient network errors.

### Query locally
```
python src/rag.py
```

---

## Budget & efficiency notes
- Embeddings are always sent in **batches**, never one-by-one.
- The corpus is embedded **once**; hyperparameter tuning used a small 200-article subset in separate namespaces to avoid re-embedding.
- The relevance filter and per-query-type retrieval avoid pushing unnecessary data into the model context.

## Deployment
Deployed on Vercel. The `api/` directory exposes the two serverless endpoints; `src/` holds the RAG logic. The dataset CSV is **not** committed or deployed — the live app queries Pinecone, not the CSV.
