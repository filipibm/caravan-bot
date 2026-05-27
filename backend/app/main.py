from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .chat import answer
from .retriever import retrieve

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["POST"],
    allow_headers=["*"],
)

OUT_OF_SCOPE = (
    "I can only answer questions about Teknik Karavan products and services. "
    "For anything else, please contact the dealer directly at info@teknikkaravan.com.tr or +90 444 79 50."
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    chunks, top_score = await retrieve(req.question)
    if not chunks:
        return ChatResponse(answer=OUT_OF_SCOPE, sources=[])
    reply = answer(req.question, chunks)
    return ChatResponse(answer=reply, sources=chunks)
