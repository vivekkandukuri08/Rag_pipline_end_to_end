# End-to-End RAG Chatbot

An end-to-end Retrieval-Augmented Generation (RAG) chatbot built using FastAPI, Sentence Transformers, Pinecone Vector Database, and Google's Gemini API.

## Features

- Uploads and processes PDF documents
- Splits text into chunks
- Generates embeddings using Sentence Transformers
- Stores embeddings in Pinecone
- Retrieves relevant context
- Generates answers using Gemini 2.5 Flash
- FastAPI web interface
- Docker support

## Tech Stack

- Python
- FastAPI
- Pinecone
- Gemini API
- Sentence Transformers
- Docker
- HTML/CSS

## Installation

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

or

```bash
python -m uvicorn main:app --reload
```

or Docker

```bash
docker build -t rag-chatbot .
docker run -p 8000:8000 --env-file .env rag-chatbot
```

Open:

```
http://localhost:8000
```

## Project Structure

```
data/
embeddings/
llm/
loaders/
splitters/
templates/
vectorstores/
main.py
config.py
requirements.txt
dockerfile
```

## Author

Madhukar Bommera
