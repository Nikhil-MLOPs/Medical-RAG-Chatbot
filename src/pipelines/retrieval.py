from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.utils.logger import logger

CHROMA_PATH = "data/chroma_db"

_vectorstore = None
_retriever = None


def _load_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        logger.info("Loading Chroma vector store (one-time)...")

        embeddings = OllamaEmbeddings(model="mxbai-embed-large")

        _vectorstore = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )

        logger.info("Chroma vector store loaded and cached")

    return _vectorstore


def get_retriever(k: int = 4):
    global _retriever

    if _retriever is None:
        logger.info("Initializing retriever (one-time)...")

        vectorstore = _load_vectorstore()
        _retriever = vectorstore.as_retriever(search_kwargs={"k": k})

        logger.info("Retriever initialized and cached")

    return _retriever
