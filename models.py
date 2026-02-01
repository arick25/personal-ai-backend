import os
import requests

HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def generate_text(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True,
        }
    }
    resp = requests.post(HUGGINGFACE_API_URL, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
        return data[0]["generated_text"]

    return str(data)
