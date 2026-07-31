# DocMind — Ask Your Documents

A lightweight Retrieval-Augmented Generation (RAG) app. Upload a PDF, ask a
question, and get an answer grounded in the document's actual content —
not a hallucinated guess.

## How it works

1. **Extraction** — `PyPDF2` pulls raw text out of the uploaded PDF.
2. **Chunking** — the text is split into overlapping ~400-word chunks. Overlap
   prevents an answer's supporting evidence from being split across a chunk
   boundary.
3. **Retrieval** — chunks are vectorized with **TF-IDF** (`scikit-learn`).
   When a question comes in, it's vectorized the same way and compared to
   every chunk via **cosine similarity**. The top 3 most relevant chunks are
   selected.
4. **Generation** — the retrieved chunks are passed to an LLM (`openai/gpt-oss-120b`,
   served via Groq's free API) as context, with an explicit instruction to
   answer only from that context (grounded generation — this is what keeps
   the answers honest instead of hallucinated).

This is the same retrieve-then-generate pattern used in production RAG
systems, scoped down to a single-file, dependency-light implementation using
TF-IDF instead of dense vector embeddings, and running entirely on free-tier
infrastructure.

## Tech stack

Python · Streamlit · PyPDF2 · scikit-learn (TF-IDF + cosine similarity) · Groq API (LLM inference, free tier)

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then paste your Groq API key into the sidebar. Get one free, no credit card
required, at [console.groq.com/keys](https://console.groq.com/keys) —
sign up with email or Google, takes under a minute.

## Possible extensions

- Swap TF-IDF for dense embeddings (`sentence-transformers` + FAISS/Chroma)
  for semantic (not just keyword) retrieval
- Support multiple PDFs / a persistent document library
- Add conversation memory for follow-up questions
- Show inline citations linking each answer sentence to its source chunk
- Swap the generation model to Claude/GPT-4 once budget allows, for a
  quality comparison against the free open-weight model

## Why TF-IDF and not embeddings?

TF-IDF retrieval is fast, has zero model-download overhead, and is fully
explainable — it matches on term frequency and rarity, not learned semantic
similarity. It's a legitimate and common baseline retrieval method, and the
architecture is designed so the retrieval step could be swapped for dense
embeddings without touching the rest of the pipeline.

## Why Groq?

Groq offers a genuinely free API tier (no credit card, generous rate limits)
serving open-weight models like `openai/gpt-oss-120b` at very high inference
speed on their custom LPU hardware. This keeps the entire project runnable
at zero cost while still using a real, capable LLM for generation.
