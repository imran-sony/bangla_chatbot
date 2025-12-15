#pip install fastapi uvicorn faiss-cpu sentence-transformers groq python-dotenv

from fastapi import FastAPI
from pydantic import BaseModel
from vector_store import search_faq
from groq_llm import generate_answer

app = FastAPI(
    title="Bangla Chatbot"
)

class ChatRequest(BaseModel):
    topic: str
    difficulty: str | None = None
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    retrieved = search_faq(
        request.question,
        request.topic,
        request.difficulty
    )

    if not retrieved:
        return {"answer": "দুঃখিত, এই প্রশ্নের উত্তর পাওয়া যায়নি।"}

    context = "\n".join([doc["answer"] for doc in retrieved])
    answer = generate_answer(context, request.question)

    return {"answer": answer}
