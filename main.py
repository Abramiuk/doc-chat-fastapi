from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Context-Aware AI Assistant (RAG)")

@app.get("/")
def read_root():
    return {"message": "Backend is running! Go to /docs to test the API."}

# Новий ендпоінт для завантаження документів
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Читаємо вміст файлу
    content = await file.read()
    
    # Декодуємо байти в звичайний текст
    text = content.decode("utf-8")
    
    # Повертаємо статистику
    return {
        "filename": file.filename, 
        "total_characters": len(text),
        "message": "File successfully received by the server!"
    }