from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from src.utils.logger import logger
from src.utils.session_store import SessionStore
from src.pipelines.rag_chain import (
    build_rag_answer,
    stream_rag_answer,
    stream_chat_answer,
)

# -----------------------------------
# GLOBAL SESSION STORE
# -----------------------------------
session_store = None


# -----------------------------------
# LIFESPAN (STARTUP / SHUTDOWN)
# -----------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global session_store
    try:
        logger.info("Starting application... Initializing Redis...")
        session_store = SessionStore()
        logger.info("Redis Session Store Ready")
    except Exception as e:
        logger.error(f"Redis initialization failed: {e}")
        session_store = None

    yield

    logger.info("Shutting down application...")


# -----------------------------------
# FASTAPI APP
# -----------------------------------
app = FastAPI(
    lifespan=lifespan,
    title="Medical RAG Chatbot API",
    version="1.0.0",
    description="Reliable medical chatbot backed by RAG, Ollama, Chroma",
)


# -----------------------------------
# REQUEST MODELS
# -----------------------------------
class QueryRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


# -----------------------------------
# HEALTH
# -----------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Medical RAG API running"}


# -----------------------------------
# ASK (NON-STREAMING, JSON)
# -----------------------------------
@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        logger.info(f"[ASK] Question: {request.question}")

        result = build_rag_answer(
            question=request.question,
            k=4,
        )

        return result

    except Exception as e:
        logger.error(f"/ask failed: {str(e)}")
        raise HTTPException(500, "Medical answer generation failed")


# -----------------------------------
# ASK STREAM (TEXT ONLY)
# -----------------------------------
@app.post("/ask-stream")
def ask_stream(request: QueryRequest):
    try:
        logger.info(f"[ASK-STREAM] Question: {request.question}")

        def event_generator():
            for chunk in stream_rag_answer(
                question=request.question,
                k=4,
            ):
                yield chunk

        return StreamingResponse(
            event_generator(),
            media_type="text/plain",
        )

    except Exception as e:
        logger.error(f"/ask-stream failed: {str(e)}")
        raise HTTPException(500, "Streaming failed")


# -----------------------------------
# CHAT (STREAMING ONLY — NO JSON EVER)
# -----------------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    if session_store is None:
        raise HTTPException(500, "Session service not ready")

    session_id = request.session_id
    question = request.question
    history = session_store.get_history(session_id)

    logger.info(f"[CHAT] Session: {session_id}")
    logger.info(f"[CHAT] Question: {question}")

    def event_generator():
        try:
            for chunk in stream_chat_answer(
                question=question,
                history=history,
                k=4,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"[CHAT STREAM ERROR]: {e}")
            yield "\n\n[ERROR] Chat streaming failed."

    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
    )
