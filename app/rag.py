from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

VECTOR_STORE_PATH = "vectorstore"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=75)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("PDF appears empty or unreadable.")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return len(chunks)

def retrieve(query: str, k: int = 6):
    if not os.path.exists(VECTOR_STORE_PATH):
        return []

    vectorstore = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    results = vectorstore.similarity_search(query, k=k)

    if not results:
        return []

    return [r.page_content for r in results]
