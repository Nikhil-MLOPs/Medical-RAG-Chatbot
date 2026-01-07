from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from src.pipelines.retrieval import get_retriever
from src.utils.logger import logger
from src.utils.exceptions import RAGError


def get_llm():
    try:
        logger.info("Initializing Ollama LLM...")

        llm = ChatOllama(
            model="mistral",   # can switch later
            temperature=0.1   # low temp reduces hallucination
        )

        logger.info("LLM initialized successfully.")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise RAGError("LLM initialization failed")


RAG_PROMPT = """
You are a highly reliable and cautious Medical AI Assistant.
Use ONLY the provided context to answer the question.

If the answer is not clearly present in the context, reply:
"I cannot answer this based on the provided medical reference."

Rules:
- Do NOT guess
- Do NOT fabricate medical facts
- Cite the page number and source file whenever possible
- Be concise but medically correct

Question:
{question}

Context:
{context}

Answer:
"""


def build_rag_chain(k=5):
    try:
        retriever = get_retriever(k=k)
        llm = get_llm()

        prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

        # Parallel data flow:
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
        )

        logger.info("RAG pipeline constructed successfully")
        return chain

    except Exception as e:
        logger.error(f"Failed to build RAG chain: {e}")
        raise RAGError("RAG chain creation failed")