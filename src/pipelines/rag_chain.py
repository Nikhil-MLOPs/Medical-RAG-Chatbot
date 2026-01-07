import time
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

        t1 = time.time()
        docs = retriever.invoke(question)
        retrieval_time = round(time.time() - t1, 3)

        logger.info(f"Retrieved {len(docs)} docs in {retrieval_time}s")

        context_text = ""
        sources = []
        previews = []

        for doc in docs:
            page = doc.metadata.get("page", "unknown")
            src = doc.metadata.get("source", "unknown")

            sources.append({"source": src, "page": page})
            previews.append(doc.page_content[:250])

            context_text += f"\n\n[Page {page}] {doc.page_content}"

        prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        llm = get_llm()

        chain = prompt | llm

        t2 = time.time()
        answer = chain.invoke({"question": question, "context": context_text})
        generation_time = round(time.time() - t2, 3)

        total_time = retrieval_time + generation_time

        return {
            "answer": answer.content,
            "sources": sources,
            "retrieval_preview": previews,
            "timing": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": round(total_time, 3)
            }
        }

    except Exception as e:
        logger.error(f"RAG failure: {str(e)}")
        raise RAGError("RAG execution failed")
    
def stream_rag_answer(question: str, k: int = 4):
    try:
        retriever = get_retriever(k=k)

        t1 = time.time()
        docs = retriever.invoke(question)
        retrieval_time = round(time.time() - t1, 3)

        logger.info(f"[STREAM] Retrieved {len(docs)} docs in {retrieval_time}s")

        context_text = ""
        sources = []

        for doc in docs:
            page = doc.metadata.get("page", "unknown")
            src = doc.metadata.get("source", "unknown")

            sources.append({"source": src, "page": page})
            context_text += f"\n\n[Page {page}] {doc.page_content}"

        prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
        llm = get_llm()

        chain = prompt | llm

        # Streaming begins
        t2 = time.time()
        for chunk in chain.stream({"question": question, "context": context_text}):
            yield chunk.content

        generation_time = round(time.time() - t2, 3)
        total = retrieval_time + generation_time

        yield f"\n\n[SOURCES]: {sources}"
        yield f"\n[TIMING]: retrieval={retrieval_time}s, llm={generation_time}s, total={total}s"

    except Exception as e:
        logger.error(f"Streaming RAG failed: {str(e)}")
        yield "Streaming failed due to an internal error."

CHAT_PROMPT = """
You are a highly reliable and cautious Medical AI Assistant.

You are in a conversation with a user.
You must use:
1) The conversation history
2) The retrieved medical context

Rules:
- Use ONLY medically grounded information from the context.
- DO NOT invent medical facts.
- If answer is not present in context, say:
  "I cannot answer this based on the provided medical reference."
- Keep responses concise and medically clear.

Conversation History:
{history}

User Question:
{question}

Medical Context:
{context}

Answer:
"""

def build_chat_answer(question: str, history: list, k: int = 4):
    try:
        retriever = get_retriever(k=k)

        t1 = time.time()
        docs = retriever.invoke(question)
        retrieval_time = round(time.time() - t1, 3)

        logger.info(f"[CHAT] Retrieved {len(docs)} docs in {retrieval_time}s")

        context_text = ""
        sources = []
        previews = []

        for doc in docs:
            page = doc.metadata.get("page", "unknown")
            src = doc.metadata.get("source", "unknown")

            sources.append({"source": src, "page": page})
            previews.append(doc.page_content[:250])
            context_text += f"\n\n[Page {page}] {doc.page_content}"

        history_text = ""
        for turn in history:
            history_text += f"\n{turn['role'].upper()}: {turn['message']}"

        prompt = ChatPromptTemplate.from_template(CHAT_PROMPT)
        llm = get_llm()

        chain = prompt | llm

        t2 = time.time()
        response = chain.invoke({
            "history": history_text,
            "question": question,
            "context": context_text
        })

        generation_time = round(time.time() - t2, 3)
        total = retrieval_time + generation_time

        return {
            "answer": response.content,
            "sources": sources,
            "retrieval_preview": previews,
            "timing": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": round(total, 3)
            }
        }

    except Exception as e:
        logger.error(f"Chat RAG failed: {str(e)}")
        raise RAGError("Chat mode RAG failed")