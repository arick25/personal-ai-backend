from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ingest import ingest_file_content
from chat import chat_with_knowledge

app = FastAPI(title="Personal RAG AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)):
    content_bytes = await file.read()
    content = content_bytes.decode("utf-8", errors="ignore")
    ingest_file_content(file.filename, content)
    return {"status": "ok", "filename": file.filename}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    reply = chat_with_knowledge(payload.message)
    return ChatResponse(reply=reply)

@app.get("/health")
async def health():
    return {"status": "ok"}
