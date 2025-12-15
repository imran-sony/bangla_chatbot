# Bangla Chatbot with FAISS & Groq LLM

A Bangla question-answering chatbot using FastAPI, FAISS for retrieval, and Groq LLM for natural language generation.  
It supports topic & difficulty filtering and provides fallback responses if an answer is not found.

## Features

🔹 Search FAQs by topic and difficulty.

🔹 Retrieval-Augmented Generation (RAG) using Groq LLM for natural Bangla answers.

🔹 FAISS for fast semantic search.

🔹 Fallback response if no FAQ matches.

🔹 Simple to use with FastAPI.

## Project Structure
bangla-chatbot/  
├─ main.py              # FastAPI server  
├─ vector_store.py      # FAISS search & embedding  
├─ groq_llm.py          # Groq LLM integration  
├─ faq_data.py          # Sample FAQ dataset  
├─ .env                 # Groq API key  
├─ requirements.txt     # Optional: dependencies  
└─ README.md  


## Installation

Clone the repository:
```bash
git clone https://github.com/imran-sony/bangla_chatbot.git  
cd bangla_chatbot
```

## Create a virtual environment

python -m venv venv
venv\Scripts\activate     # Windows


## Install dependencies

pip install fastapi uvicorn faiss-cpu sentence-transformers groq python-dotenv

Setup

Create a .env file in the root folder:

GROQ_API_KEY=groq_api_key


## Usage

Run the FastAPI server:

uvicorn main:app --reload --host 0.0.0.0 --port 8000

Access API documentation: http://127.0.0.1:8000/docs
