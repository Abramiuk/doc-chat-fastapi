from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Context-Aware AI Assistant (RAG)")

# --- ЛОГІКА RAG ---
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Ріже текст на шматки заданого розміру з невеликим нахлестом.
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        # Зсуваємо старт не на повний chunk_size, а залишаємо overlap (нахлест)
        start += chunk_size - overlap 
        
    return chunks

# --- ENDPOINTS (МАРШРУТИ) ---
@app.get("/")
def read_root():
    return {"message": "Backend is running! Go to /docs to test the API."}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Читаємо файл
    content = await file.read()
    text = content.decode("utf-8")
    
    # 2. Ріжемо текст на чанки
    text_chunks = chunk_text(text, chunk_size=500, overlap=50)
    
    # 3. Віддаємо статистику фронтенду
    return {
        "filename": file.filename, 
        "total_characters": len(text),
        "total_chunks": len(text_chunks),
        "first_chunk_preview": text_chunks[0][:100] + "..." if text_chunks else "",
        "message": "File successfully received and chunked!"
    }