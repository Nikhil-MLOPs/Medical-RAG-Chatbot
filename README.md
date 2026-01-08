**Medical RAG Chatbot (Local-First, Streaming)**

This project is a local-first, production-shaped Medical RAG Chatbot built on top of a medical textbook PDF.
It is designed to be reliable, safe, transparent, and cost-free, while still following real backend and ML engineering practices.

The system uses Retrieval-Augmented Generation (RAG) to answer medical questions strictly from provided medical references, avoiding hallucinations and unsupported claims. All inference runs locally using Ollama.

**What This Project Solves**

Medical question answering is a high-risk domain. This system is intentionally designed to:

- Answer only from the medical source PDF

- Refuse to guess when information is not present

- Show retrieval and generation timings transparently

- Provide a streaming conversational experience

- Work fully offline and locally, without cloud costs

- It is a complete end-to-end system.

**Architecture Overview**

The system is split cleanly into backend and frontend:

- FastAPI serves as the backend API

- Streamlit provides a ChatGPT-style streaming UI

- ChromaDB stores vector embeddings of the medical PDF

- Ollama runs local embedding and LLM inference

- Redis (optional, local) stores conversational memory

- MLflow tracks evaluation metrics and experiments

- Docker + CI provide reproducibility and stability

- The frontend never blocks on long inference. All answers are streamed token-by-token from the backend.

**Key Features**

- Local-first (no paid APIs, no cloud dependency)

- Streaming chat UI by default

- Conversational memory using session IDs

- Medical safety guardrails in prompt design

- Transparent performance metrics (retrieval, generation, total)

- Batch and single-query evaluation with MLflow

- Clean project structure with logging and exception handling

- Dockerized and CI-validated

- Project Structure

medical_rag_chatbot/
│
├── app/
│   ├── main.py              # FastAPI backend
│   └── streamlit_app.py     # Streamlit chat UI
│
├── src/
│   ├── pipelines/
│   │   ├── ingestion.py     # PDF loading + chunking
│   │   ├── retrieval.py     # Chroma retriever (cached)
│   │   └── rag_chain.py     # RAG + streaming logic
│   │
│   ├── utils/
│   │   ├── logger.py
│   │   ├── exceptions.py
│   │   └── session_store.py # Redis session memory
│   │
│   └── experiments/
│       └── mlflow_manager.py
│
├── data/
│   └── chroma_db/           # Vector store (DVC tracked)
│
├── scripts/
│   └── evaluation/          # Batch and single evaluation
│
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md

**How the System Works**

- A medical PDF is ingested and chunked.

- Chunks are embedded using a local Ollama embedding model.

- Embeddings are stored in ChromaDB.

- A user asks a question through the Streamlit UI.

- FastAPI retrieves the most relevant chunks.

- The LLM generates an answer only from retrieved context.

- Tokens are streamed back to the UI.

- Retrieval time, LLM time, and total time are displayed.

**Running the Project Locally - Prerequisites**

- Python 3.12+

- Ollama installed and running locally

- mistral and mxbai-embed-large pulled in Ollama

- Redis (optional, only for chat memory)

- Create Environment -> uv sync

- Start Backend (Terminal 1) -> uv run uvicorn app.main:app --reload

- Start Frontend (Terminal 2) -> uv run streamlit run app/streamlit_app.py

- Then open the Streamlit URL shown in the terminal.

**Evaluation & Experiments**

- Evaluation is a first-class concern in this project.

- Single-query evaluation logs latency metrics

- Batch evaluation runs across a question set

- MLflow tracks: Retrieval time, Generation time, Total time, Aggregate statistics

- MLflow uses a SQLite backend for stability and future-proofing.

**Design Decisions (Intentional)**

- Local-only inference: avoids cost, ensures privacy

- Streaming by default: long inference should feel responsive

- Context length control: prevents runaway latency

- Cached vector store and LLM clients: avoids repeated cold starts

- No hallucination tolerance: explicit refusal when context is missing

**What This Project Is (and Is Not)**

This project is:

- A realistic, production-shaped RAG system

- A demonstration of end-to-end ML + backend engineering

This project is not:

- A cloud-scale SaaS

- A GPU-optimized inference system

- A medical advice replacement

**Final Note**

This system was built step-by-step with correctness, safety, and engineering discipline as priorities.