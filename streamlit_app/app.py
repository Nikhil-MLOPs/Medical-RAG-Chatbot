import streamlit as st
import requests
import uuid
import re

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
FASTAPI_CHAT_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical RAG Chatbot")
st.caption(
    "Medical answers grounded strictly in textbook references. "
    "No assumptions. No hallucinations."
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# RENDER CHAT HISTORY
# -------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander("📚 Medical References"):
                for src in msg["sources"]:
                    st.markdown(
                        f"- **Medical Textbook** — Page `{src['page']}`"
                    )

        if msg.get("timing"):
            st.caption(
                f"⏱ Retriever: {msg['timing']['retrieval']}s · "
                f"LLM: {msg['timing']['llm']}s · "
                f"Total: {msg['timing']['total']}s"
            )

# -------------------------------------------------
# USER INPUT
# -------------------------------------------------
user_input = st.chat_input("Ask a medical question…")

if user_input:
    # ---- USER MESSAGE ----
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # ---- ASSISTANT MESSAGE (STREAMING) ----
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()

        full_answer = ""
        sources = []
        timing = None

        try:
            with requests.post(
                FASTAPI_CHAT_URL,
                json={
                    "session_id": st.session_state.session_id,
                    "question": user_input
                },
                stream=True,
                timeout=None
            ) as response:

                if response.status_code != 200:
                    answer_placeholder.error(
                        "Medical answer generation failed."
                    )
                else:
                    for chunk in response.iter_content(chunk_size=1024):
                        if not chunk:
                            continue

                        text = chunk.decode("utf-8")

                        # ---- TIMING BLOCK ----
                        if text.startswith("[TIMING]"):
                            match = re.search(
                                r"retrieval=(\d+\.?\d*)s, llm=(\d+\.?\d*)s, total=(\d+\.?\d*)s",
                                text
                            )
                            if match:
                                timing = {
                                    "retrieval": match.group(1),
                                    "llm": match.group(2),
                                    "total": match.group(3)
                                }
                            continue

                        # ---- SOURCES BLOCK ----
                        if text.startswith("[SOURCES]"):
                            try:
                                sources = eval(
                                    text.replace("[SOURCES]:", "").strip()
                                )
                            except Exception:
                                sources = []
                            continue

                        # ---- STREAM ANSWER TEXT ----
                        full_answer += text
                        answer_placeholder.markdown(full_answer)

        except Exception as e:
            answer_placeholder.error(f"Connection failed: {e}")

    # ---- SAVE ASSISTANT MESSAGE ----
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
            "timing": timing
        }
    )