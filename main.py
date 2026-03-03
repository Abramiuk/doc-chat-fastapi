from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import chromadb
import uuid
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- LOAD SECRET API KEY ---
load_dotenv() # This reads the .env file
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("API Key not found! Please check your .env file.")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel('gemini-flash-latest') 

app = FastAPI(title="Context-Aware AI Assistant (RAG)")

# --- DATABASE INITIALIZATION ---
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

class QueryRequest(BaseModel):
    question: str
    n_results: int = 2

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

@app.post("/ask")
async def ask_assistant(query: QueryRequest):
    try:
        # 1. RETRIEVAL
        results = collection.query(
            query_texts=[query.question],
            n_results=query.n_results
        )
        
        if not results["documents"] or not results["documents"][0]:
            raise HTTPException(status_code=404, detail="No documents found in the database.")
        
        found_context = " ".join(results["documents"][0])
        
        # 2. AUGMENTATION
        prompt = f"""
        You are a smart and concise assistant. Your task is to answer the user's question.
        You MUST base your answer EXCLUSIVELY on the following document context.
        If the answer is not in the context, simply say: "The uploaded document does not contain an answer to this question."
        
        DOCUMENT CONTEXT:
        {found_context}
        
        USER QUESTION: 
        {query.question}
        """
        
        # 3. GENERATION
        response = llm.generate_content(prompt)
        
        return {
            "question": query.question,
            "answer": response.text, 
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")