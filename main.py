from fastapi import FastAPI, UploadFile, File
import chromadb
import uuid

app = FastAPI(title="Context-Aware AI Assistant (RAG)")

# --- ІНІЦІАЛІЗАЦІЯ БАЗИ ДАНИХ ---
# Створюємо локальну базу, яка збережеться у папці "chroma_db" прямо у твоєму проєкті
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Створюємо або отримуємо колекцію (це як таблиця в SQL)
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

# --- ENDPOINTS ---
@app.get("/")
def read_root():
    return {"message": "Backend is running! Go to /docs to test the API."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    
    text_chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    # 1. Генеруємо унікальні ID для кожного шматка тексту (це вимога бази)
    chunk_ids = [str(uuid.uuid4()) for _ in text_chunks]
    
    # 2. Зберігаємо метадані (щоб знати, з якого файлу цей шматок)
    metadatas = [{"source": file.filename} for _ in text_chunks]
    
    # 3. МАГІЯ: Додаємо в базу! 
    # ChromaDB автоматично перетворить текст на вектори (Embeddings) під капотом
    collection.add(
        documents=text_chunks,
        ids=chunk_ids,
        metadatas=metadatas
    )
    
    return {
        "filename": file.filename, 
        "total_chunks_saved": len(text_chunks),
        "db_status": f"Successfully embedded and saved to ChromaDB collection: {collection.name}"
    }