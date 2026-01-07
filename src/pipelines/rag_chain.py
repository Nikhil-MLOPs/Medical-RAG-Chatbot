from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from src.pipelines.retrieval import get_retriever
from src.utils.logger import logger
from src.utils.exceptions import RAGError


RAG_PROMPT = """
You are a highly reliable and cautious Medical AI Assistant.
Use ONLY the provided medical context to answer the question.

If the answer is not clearly contained in the context,
respond:

"I cannot answer this based on the provided medical reference."

Rules:
- Do NOT guess.
- Do NOT invent medical information.
- Prefer short, precise medical explanations.
- Include referenced page numbers in your explanation when possible.

Question:
{question}

Medical Context:
{context}

Answer:
"""


def get_llm():
    try:
        logger.info("Initializing Ollama LLM...")

        llm = ChatOllama(
            model="mistral",
            temperature=0.1,
        )

        logger.info("LLM initialized successfully.")
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise RAGError("LLM initialization failed")


def build_rag_answer(question: str, k: int = 4):
    try:
        retriever = get_retriever(k=k)

        logger.info("Running retrieval...")
        docs = retriever.invoke(question)

        logger.info(f"Retrieved {len(docs)} documents")

        # Build context string
        context_text = ""
        sources = []

        for doc in docs:
            page = doc.metadata.get("page", "unknown")
            src = doc.metadata.get("source", "unknown")

            sources.append({
                "source": src,
                "page": page
            })

            context_text += f"\n\n[Page {page}] {doc.page_content}"

        prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        llm = get_llm()

        chain = prompt | llm

        logger.info("Invoking LLM with grounded context...")
        answer = chain.invoke({
            "question": question,
            "context": context_text
        })

        return answer.content, sources

    except Exception as e:
        logger.error(f"RAG chain failure: {str(e)}")
        raise RAGError("Failed to run RAG pipeline")