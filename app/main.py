from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipelines.rag_chain import build_rag_chain
from src.utils.logger import logger

app = FastAPI(
    title="Medical RAG Chatbot API",
    version="1.0.0",
    description="Reliable medical chatbot backed by RAG, Ollama, Chroma, FastAPI"
)


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    sources: list


# Global chain initialization
try:
    logger.info("Initializing RAG Chain at startup...")
    rag_chain = build_rag_chain(k=4)
    logger.info("RAG Chain initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize RAG chain at startup: {str(e)}")
    rag_chain = None


@app.get("/health")
def health_check():
    return {"status": "Ok", "message": "Medical RAG API running"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not rag_chain:
        logger.error("RAG chain not initialized")
        raise HTTPException(status_code=500, detail="RAG system not ready")

    try:
        logger.info(f"User Question: {request.question}")

        response = rag_chain.invoke(request.question)

        answer_text = response.content

        # Extract metadata from the retriever step (available inside context)
        # LangChain does not directly attach docs in output, so for now we return placeholders.
        # In Phase-4, we will attach exact cited docs.

        return {
            "answer": answer_text,
            "sources": ["medical_book.pdf"]
        }

    except Exception as e:
        logger.error(f"Error while processing question: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate response")