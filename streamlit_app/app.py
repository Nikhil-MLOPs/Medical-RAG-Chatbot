import requests
import streamlit as st

# -------------------------------
# CONFIG
# -------------------------------
FASTAPI_URL = "http://127.0.0.1:8000/ask"


st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical RAG Chatbot")
st.caption("Reliable Medical Assistant | Powered by RAG + Ollama + Chroma + FastAPI")


# -------------------------------
# INPUT
# -------------------------------
question = st.text_area(
    "Ask any medical question based on our textbook:",
    height=120,
    placeholder="Example: What is diabetes and how is it defined?"
)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Contacting Medical Knowledge Base..."):
        try:
            response = requests.post(
                FASTAPI_URL,
                json={"question": question},
                timeout=None
            )

            if response.status_code != 200:
                st.error("Server error. Please try again later.")
                st.stop()

            data = response.json()

        except Exception as e:
            st.error(f"Connection failed: {e}")
            st.stop()

    # -------------------------------
    # DISPLAY ANSWER
    # -------------------------------
    st.subheader("Answer")
    st.write(data["answer"])

    st.success("AI followed medical-safe RAG pipeline successfully")

    # -------------------------------
    # SOURCES
    # -------------------------------
    st.subheader("📚 References Used")
    for src in data["sources"]:
        st.write(f"- Source: `{src['source']}` | Page: `{src['page']}`")

    # -------------------------------
    # TIMING
    # -------------------------------
    st.subheader("⚙️ Performance")
    timing = data["timing"]

    st.write(f"Retrieval Time: `{timing['retrieval_time']}s`")
    st.write(f"Generation Time: `{timing['generation_time']}s`")
    st.write(f"Total Response Time: `{timing['total_time']}s`")
