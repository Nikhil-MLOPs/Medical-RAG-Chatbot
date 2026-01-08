from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipelines.rag_chain import build_rag_answer
from src.pipelines.rag_chain import build_chat_answer
from src.utils.logger import logger

from fastapi.responses import StreamingResponse
from src.pipelines.rag_chain import stream_rag_answer

from contextlib import asynccontextmanager
from src.utils.session_store import SessionStore

# -----------------------------------
# GLOBAL SESSION STORE (lazy init)
# -----------------------------------
session_store = None


# -----------------------------------
# LIFESPAN STARTUP SHUTDOWN HANDLER
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
    description="Reliable medical chatbot backed by RAG, Ollama, Chroma"
)


# -----------------------------------
# REQUEST MODELS
# -----------------------------------
class QueryRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    session_id: str
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


# -----------------------------------
# HEALTH
# -----------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Medical RAG API running"}


# -----------------------------------
# ASK
# -----------------------------------
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


# -----------------------------------
# ASK STREAM
# -----------------------------------
@app.post("/ask-stream")
def ask_stream(request: QueryRequest):
    try:
        logger.info(f"Streaming Question: {request.question}")

        def event_generator():
            for chunk in stream_rag_answer(request.question, k=4):
                yield chunk

        return StreamingResponse(event_generator(), media_type="text/plain")

    except Exception as e:
        logger.error(f"/ask-stream failed: {str(e)}")
        raise HTTPException(500, "Streaming failed")


# -----------------------------------
# CHAT
# -----------------------------------
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        if session_store is None:
            raise HTTPException(500, "Session service not ready")

        session_id = request.session_id
        history = session_store.get_history(session_id)

        logger.info(f"[CHAT] Session: {session_id}")
        logger.info(f"[CHAT] Question: {request.question}")

        result = build_chat_answer(
            question=request.question,
            history=history,
            k=4
        )

        session_store.save_turn(
            session_id=session_id,
            user_message=request.question,
            assistant_message=result["answer"]
        )

        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "history_length": len(history) + 2,
            "timing": result["timing"]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/chat failed: {str(e)}")
        raise HTTPException(500, "Chat mode failed")
