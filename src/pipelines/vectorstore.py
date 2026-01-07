from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.pipelines.ingestion import run_ingestion_pipeline
from src.utils.logger import logger
from src.utils.exceptions import EmbeddingError


CHROMA_DB_PATH = "data/chroma_db"


def get_ollama_embedding():
    try:
        logger.info("Initializing Ollama Embedding Model...")

        embedding_model = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        logger.info("Ollama Embedding Model initialized successfully")
        return embedding_model

    except Exception as e:
        logger.error(f"Ollama Embedding initialization failed: {str(e)}")
        raise EmbeddingError("Failed to initialize Ollama Embeddings")


def create_chroma_database(chunks):
    try:
        logger.info("Creating Chroma Vector Store...")

        embeddings = get_ollama_embedding()

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )

        # ❌ DO NOT CALL persist() — auto persistence happens
        logger.info("Chroma Vector Store created and persisted successfully")

        return vector_db

    except Exception as e:
        logger.error(f"Chroma DB creation failed: {str(e)}")
        raise EmbeddingError("Failed to create Chroma Vector Store")


def run_vector_pipeline():
    logger.info("PHASE-2 VECTOR STORE PIPELINE STARTED")

    chunks = run_ingestion_pipeline()
    vector_db = create_chroma_database(chunks)

    logger.info("PHASE-2 VECTOR STORE PIPELINE COMPLETED SUCCESSFULLY")
    return vector_db