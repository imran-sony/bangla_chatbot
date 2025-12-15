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

    # If no FAQ found, return fallback
    if not retrieved:
        return {"answer": "দুঃখিত, এই প্রশ্নের উত্তর পাওয়া যায়নি।"}

    # Join answers, skip empty ones
    context = "\n".join([doc.get("answer", "") for doc in retrieved if doc.get("answer")])
    
    if not context:
        return {"answer": "দুঃখিত, এই প্রশ্নের উত্তর পাওয়া যায়নি।"}

    try:
        answer = generate_answer(context, request.question)
        return {"answer": answer}
    except Exception as e:
        # Fallback in case of LLM error
        return {"answer": f"Internal error: {str(e)}"}
