from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import logger
from src.utils.exceptions import PDFIngestionError
from configs import ingestion_config


def load_pdf_documents():
    try:
        logger.info("Starting PDF loading process...")

        loader = PyPDFLoader(ingestion_config.PDF_PATH)
        documents = loader.load()

        logger.info(f"Successfully loaded PDF. Total pages: {len(documents)}")
        return documents

    except Exception as e:
        logger.error(f"PDF loading failed: {str(e)}")
        raise PDFIngestionError("Failed to load PDF")


def split_documents(documents):
    try:
        logger.info("Starting document chunking...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=ingestion_config.CHUNK_SIZE,
            chunk_overlap=ingestion_config.CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(documents)

        # enrich metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = idx
            chunk.metadata["source"] = ingestion_config.PDF_PATH

        logger.info(f"Document chunking complete. Total chunks created: {len(chunks)}")
        return chunks

    except Exception as e:
        logger.error(f"Document chunking failed: {str(e)}")
        raise PDFIngestionError("Failed to split PDF into chunks")


def run_ingestion_pipeline():
    logger.info("PHASE-2 INGESTION PIPELINE STARTED")

    docs = load_pdf_documents()
    chunks = split_documents(docs)

    logger.info("PHASE-2 INGESTION PIPELINE COMPLETED SUCCESSFULLY")
    return chunks
