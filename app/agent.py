from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.rag import retrieve
import os

SYSTEM_PROMPT = """You are a precise research assistant. You will be given chunks of text retrieved from a document.

Rules:
- Answer ONLY from the provided context chunks
- Be specific — use exact terms, numbers, names from the context
- If the context contains a direct answer, quote or closely paraphrase it
- If the context does not contain enough information, say exactly that
- Do not add general knowledge or assumptions
- Structure your answer clearly if it has multiple parts"""

def run_agent(query: str) -> dict:
    chunks = retrieve(query)

    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded document for this query.",
            "source_chunks": [],
            "retrieval_used": False
        }

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")
    )

    context = "\n\n---\n\n".join(chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context from document:\n{context}\n\nQuestion: {query}")
    ]

    try:
        response = llm.invoke(messages)
        answer = response.content
    except Exception as e:
        answer = f"LLM call failed: {str(e)}"

    return {
        "answer": answer,
        "source_chunks": chunks,
        "retrieval_used": True
    }
