---
title: RAG Research Agent
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# RAG Research Agent

A production-style AI agent that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## Architecture

PDF Upload → LangChain Chunking → sentence-transformers Embeddings → FAISS Vector Store → Semantic Retrieval → Llama 3.3 70B → Grounded Answer

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI (Python) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| LLM | Llama 3.3 70B via Groq |
| Container | Docker |

## Known Failure Modes and Mitigations

| Failure Mode | Mitigation |
|---|---|
| Empty PDF | 422 error returned before processing |
| No relevant chunks found | LLM call skipped, explicit not-found response |
| LLM call fails | Try/except returns error string, server stays up |
| Non-PDF upload | 400 error at validation layer |
| Context overflow | Chunk size capped at 300 tokens, top-k=6 |

## Environment Variables

| Variable | Required |
|---|---|
| GROQ_API_KEY | Yes |
