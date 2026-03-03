from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import chromadb
import uuid

app = FastAPI(title="Context-Aware AI Assistant (RAG)")

# --- ІНІЦІАЛІЗАЦІЯ БАЗИ ДАНИХ ---
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents_collection")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap 
    return chunks

# --- МОДЕЛІ ДАНИХ ---
# Цей клас описує, як має виглядати запит від користувача
class QueryRequest(BaseModel):
    question: str
    n_results: int = 2  # Скільки шматків тексту ми хочемо знайти (за замовчуванням 2)

# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "Backend is running! Go to /docs to test the API."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    
    text_chunks = chunk_text(text, chunk_size=500, overlap=50)
    chunk_ids = [str(uuid.uuid4()) for _ in text_chunks]
    metadatas = [{"source": file.filename} for _ in text_chunks]
    
    collection.add(documents=text_chunks, ids=chunk_ids, metadatas=metadatas)
    
    return {"filename": file.filename, "total_chunks_saved": len(text_chunks)}

# НОВИЙ ЕНДПОІНТ: СЕМАНТИЧНИЙ ПОШУК
@app.post("/search")
async def search_document(query: QueryRequest):
    # 1. ChromaDB сама перетворює питання на вектор
    # 2. Шукає в базі найближчі математичні співпадіння
    results = collection.query(
        query_texts=[query.question],
        n_results=query.n_results
    )
    
    return {
        "question": query.question,
        "relevant_chunks": results["documents"][0],  # Знайдений текст
        "distances": results["distances"][0]         # Наскільки це точний збіг (менше число = краще)
    }