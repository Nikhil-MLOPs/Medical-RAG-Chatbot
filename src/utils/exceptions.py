class RAGBaseException(Exception):
    """Base custom exception for the project"""
    pass


class PDFIngestionError(RAGBaseException):
    """Raised when PDF reading or processing fails"""
    pass


class EmbeddingError(RAGBaseException):
    """Raised when embedding generation or vector store operations fail"""
    pass


class RetrievalError(RAGBaseException):
    """Raised when retrieval pipeline fails"""
    pass