from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.rag import retrieve
import os

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

SYSTEM_PROMPT = """You are a research assistant. You are given retrieved context chunks from a document.
Use them to answer the user's question precisely.
If the context is insufficient or empty, clearly say so — do not hallucinate an answer.
Always mention which part of the context you used."""

def run_agent(query: str) -> dict:
    chunks = retrieve(query)

    if not chunks:
        return {
            "answer": "No relevant content found in the uploaded document for this query.",
            "source_chunks": [],
            "retrieval_used": False
        }

    context = "\n\n---\n\n".join(chunks)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
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
