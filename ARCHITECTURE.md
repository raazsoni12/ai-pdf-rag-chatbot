# 🧠 System Architecture

PDF → Chunking → Embeddings (SentenceTransformers) → FAISS Vector DB  
→ Retriever → Context Injection → Mistral (Ollama) → Answer

## Flow

1. User uploads one or more PDFs
2. PDFs are split into chunks
3. Chunks converted into embeddings locally
4. FAISS stores vectors
5. Question is embedded
6. Similar chunks retrieved
7. Context sent to Mistral via Ollama
8. Answer returned with sources

