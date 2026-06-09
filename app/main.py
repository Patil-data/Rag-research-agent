from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.agent import run_agent
from app.rag import ingest_pdf
import shutil
import os

app = FastAPI(title="RAG Research Agent")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def root():
    return FileResponse("app/static/index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        chunk_count = ingest_pdf(file_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"message": "PDF ingested successfully.", "file": file.filename, "chunks_indexed": chunk_count}

@app.post("/query")
def query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = run_agent(request.query)
    return result
