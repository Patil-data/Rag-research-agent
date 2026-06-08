# RAG Research Agent

A production-style AI agent that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## Architecture
User Query → FastAPI → Agent (Orchestrator)
↓
RAG Tool (FAISS retrieval)
↓
LLM (Llama 3.3 70B via Groq)
↓
Answer + Source Chunks
## Tech Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI (Python) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, local) |
| Vector Store | FAISS (local) |
| RAG Framework | LangChain |
| LLM | Llama 3.3 70B via Groq API |
| Chunking | RecursiveCharacterTextSplitter (500 tokens, 50 overlap) |
| Container | Docker |

## API

### Upload a PDF
POST /upload
Content-Type: multipart/form-data

### Query the document
POST /query
Content-Type: application/json
{"query": "your question here"}
### Response format
```json
{
  "answer": "LLM answer citing context",
  "source_chunks": ["chunk1", "chunk2"],
  "retrieval_used": true
}
```

## Known Failure Modes and Mitigations

| Failure Mode | How it's handled |
|---|---|
| Empty or unreadable PDF | Raises 422 with clear error message |
| No relevant chunks retrieved | Returns explicit "not found" — LLM never called, no hallucination risk |
| LLM call fails | Try/except returns error string, does not crash the server |
| Non-PDF file uploaded | 400 error returned before any processing |
| Context window overflow | Chunk size capped at 500 tokens; top-k=4 limits context sent to LLM |

## Setup

```bash
git clone https://github.com/Patil-data/rag-research-agent
cd rag-research-agent
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

```bash
docker build -t rag-agent .
docker run -e GROQ_API_KEY=your_key_here -p 8000:8000 rag-agent
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| GROQ_API_KEY | Yes | Free key from console.groq.com |

