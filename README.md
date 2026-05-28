# Medium Article RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions **only** from a corpus of ~7,600 English Medium articles. Built with Next.js, Pinecone, and OpenAI-compatible models, deployed on Vercel.

## How It Works


Question → embed → search Pinecone → retrieve top chunks
        → build augmented prompt → gpt-5-mini answers using ONLY that context

1. **Embed** the question with `text-embedding-3-small` (1536 dims).
2. **Retrieve** the most semantically similar chunks from Pinecone (cosine similarity).
3. **De-duplicate** by article (keep the best chunk per article) so list-type questions return *distinct* articles.
4. **Augment** a prompt with the retrieved passages.
5. **Generate** the answer with `gpt-5-mini`, constrained to use only the provided context.

## Chosen Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `chunk_size` | 512 words | Best balance of retrieval quality vs. context cost |
| `overlap_ratio` | 0.2 | Preserves context across chunk boundaries without excessive duplication |
| `top_k` | 5 | Enough context for grounded answers without bloating the prompt |

### Why these settings (experiment summary)

I compared three configurations on a 100-article subset, embedding each into a
separate Pinecone namespace and querying all of them with the four assignment
question types:

| Config | Q1 top1 | Q2 top1 | Q3 top1 | Q4 top1 |
|--------|---------|---------|---------|---------|
| 512 / 0.20  | 0.6314 | 0.5465 | 0.6585 | 0.5555 |
| 256 / 0.15  | 0.6130 | 0.5462 | 0.6301 | 0.5498 |
| 1024 / 0.30 | 0.6366 | 0.5485 | 0.6585 | 0.5788 |

**Findings:** All three configs retrieved the correct article at rank #1 for every
question, so retrieval is robust. `1024/0.30` had marginally higher top-1 scores but
pushes far more tokens into the model context per query (higher cost). `512/0.20` matched it closely on quality
while using roughly half the context, so i chose **512 / 0.20** as the best
quality-to-cost tradeoff.

## API Endpoints

### `POST /api/prompt`
**Input:** `{ "question": "..." }`
**Output:** `{ "response": ..., "context": [...], "augmented_prompt": { "System": ..., "User": ... } }`

### `GET /api/stats`
**Output:** `{ "chunk_size": 512, "overlap_ratio": 0.2, "top_k": 5 }`

## Cost Control

- Started with a 100-article subset to validate the full pipeline before scaling.
- Tuned hyperparameters on the subset (cheap) rather than the full corpus.
- Ingestion uses a progress file to resume after interruptions without re-embedding.
- Eval measures retrieval only (no expensive chat-model calls).

## Tech Stack

- **Next.js** (App Router) — API + deployment
- **Pinecone** — vector database (cosine, 1536 dims)
- **OpenAI-compatible API** — `text-embedding-3-small` + `gpt-5-mini`
- **Vercel** — hosting
