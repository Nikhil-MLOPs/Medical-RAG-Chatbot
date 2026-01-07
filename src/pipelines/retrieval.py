from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.utils.logger import logger
from src.utils.exceptions import RetrievalError

CHROMA_DB_PATH = "data/chroma_db"


def load_vector_db():
    try:
        logger.info("Loading existing Chroma database...")

        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        db = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )

        logger.info("Chroma DB loaded successfully.")
        return db

    except Exception as e:
        logger.error(f"Failed to load Chroma DB: {e}")
        raise RetrievalError("Unable to load vector database")


def get_retriever(k=5):
    db = load_vector_db()
    retriever = db.as_retriever(search_kwargs={"k": k})
    logger.info(f"Retriever initialized with k={k}")
    return retriever