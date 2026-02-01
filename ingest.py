from typing import List
from vector_store import add_documents

def ingest_texts(texts: List[str], sources: List[str] | None = None):
    metadatas = None
    if sources:
        metadatas = [{"source": s} for s in sources]
    add_documents(texts, metadatas=metadatas)

def ingest_file_content(filename: str, content: str):
    ingest_texts([content], [filename])
