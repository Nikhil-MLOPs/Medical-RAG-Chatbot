from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipelines.rag_chain import build_rag_answer
from src.utils.logger import logger


app = FastAPI(
    title="Medical RAG Chatbot API",
    version="1.0.0",
    description="Reliable medical chatbot backed by RAG, Ollama, Chroma"
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Medical RAG API running"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        logger.info(f"Incoming Question: {request.question}")

        result = build_rag_answer(
            question=request.question,
            k=4
        )

        return result

    except Exception as e:
        logger.error(f"/ask failed: {str(e)}")
        raise HTTPException(500, "Medical answer generation failed")