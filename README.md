# Local RAG System Usage Guide

Your local RAG system is set up! Here is how to use it.

## 1. Add Resumes
Place your PDF resume files into the `input` folder.
- `c:\Users\athul\OneDrive\Desktop\projects\python\rag\input`

## 2. Ingest Data
Run the ingestion script to process the resumes and create the vector database.
```bash
venv\Scripts\python ingest.py
```
This will create a `vector_db` folder containing the embeddings.

## 3. Ask Questions
Run the query script with your question.
```bash
venv\Scripts\python query.py "What is the best candidate for backend role?"
```

## Tips
- Ensure you have the `llama3.2:3b` model running or available in Ollama (`ollama list`).
- If you add new resumes, run `ingest.py` again.
