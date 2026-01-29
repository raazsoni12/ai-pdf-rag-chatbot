from fastapi import FastAPI, UploadFile, File
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from uuid import uuid4

app = FastAPI()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Each chunk: {"text":..., "source":..., "page":...}
documents = []
index = None

def embed(texts):
    return embedder.encode(texts).astype("float32")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global documents, index

    reader = PdfReader(file.file)

    new_docs = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]

        for c in chunks:
            new_docs.append({
                "id": str(uuid4()),
                "text": c,
                "source": file.filename,
                "page": page_num + 1
            })

    documents.extend(new_docs)

    vectors = embed([d["text"] for d in documents])
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return {"status": f"{file.filename} indexed successfully"}

@app.post("/ask")
def ask(payload: dict):
    global documents, index

    if index is None:
        return {"answer": "Upload PDFs first.", "sources": []}

    question = payload.get("question", "")
    history = payload.get("history", [])

    q_vec = embed([question])
    _, I = index.search(q_vec, 3)

    contexts = [documents[i] for i in I[0]]

    context_text = "\n".join([c["text"] for c in contexts])

    chat_history = "\n".join(history)

    prompt = f"""
You are an assistant answering from documents.

Previous conversation:
{chat_history}

Context:
{context_text}

Question: {question}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"]

    sources = [{
        "file": c["source"],
        "page": c["page"]
    } for c in contexts]

    return {"answer": answer, "sources": sources}








