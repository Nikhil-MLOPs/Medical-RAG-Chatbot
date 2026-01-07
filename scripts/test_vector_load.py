from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db = Chroma(
    persist_directory="data/chroma_db",
    embedding_function=embeddings
)

print("Total docs:", db._collection.count())