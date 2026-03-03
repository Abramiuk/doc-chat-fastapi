# Context-Aware RAG API (FastAPI + Gemini)

A blazingly fast backend service for Retrieval-Augmented Generation (RAG) built with Python and FastAPI. This API allows users to upload custom documents, convert text into mathematical vector embeddings, and perform context-aware Q&A using Google's Gemini LLM.

## Features
- **FastAPI Backend:** High-performance asynchronous API with auto-generated Swagger UI documentation.
- **Document Processing:** Endpoint to securely upload and decode custom text files (`.txt`).
- **Vector Storage (ChromaDB):** Automated text chunking (500 characters with 50-character overlap) and local vector embedding storage.
- **Semantic Search:** Retrieves the most relevant document chunks based on the user's query.
- **AI Generation (Gemini):** Integrates with Google's Gemini API to generate accurate, natural-language answers strictly based on the provided document context.

## Tech Stack
- **Framework:** FastAPI, Uvicorn
- **Vector Database:** ChromaDB
- **LLM:** Google Generative AI (Gemini Pro)
- **Environment Management:** python-dotenv

