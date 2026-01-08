from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.utils.logger import logger

CHROMA_DB_PATH = "data/chroma_db"

retriever_instance = None  # <-- GLOBAL SINGLETON


def get_retriever(k: int = 4):
    global retriever_instance

    if retriever_instance is not None:
        return retriever_instance

    try:
        logger.info("Loading existing Chroma database...")

        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )

        retriever_instance = db.as_retriever(
            search_kwargs={"k": k}
        )

        logger.info("Chroma DB loaded successfully.")
        logger.info(f"Retriever initialized with k={k}")

        return retriever_instance

    except Exception as e:
        logger.error(f"Retriever Initialization Failed: {str(e)}")
        raise