from typing import List, Tuple
from vector_store import query_documents
from models import generate_text

SYSTEM_PROMPT = """You are a blank-slate assistant whose entire knowledge and personality come only from the user's provided documents.
Use ONLY the retrieved context below to answer. If the answer is not in the context, say you don't know.
Be concise and grounded in the provided text.
"""

def build_prompt(user_message: str, context_docs: List[Tuple[str, dict]]) -> str:
    context_strs = []
    for doc, meta in context_docs:
        src = meta.get("source") if meta else "unknown"
        context_strs.append(f"[Source: {src}]\n{doc}")

    context_block = "\n\n---\n\n".join(context_strs) if context_strs else "No context available."

    prompt = f"""{SYSTEM_PROMPT}

Context:
{context_block}

User: {user_message}
Assistant:"""
    return prompt

def chat_with_knowledge(user_message: str) -> str:
    docs = query_documents(user_message, n_results=5)
    prompt = build_prompt(user_message, docs)
    raw_output = generate_text(prompt)

    if "Assistant:" in raw_output:
        raw_output = raw_output.split("Assistant:", 1)[-1].strip()

    return raw_output.strip()
