import os
import streamlit as st

from graph.workflow import route_query
from rag.pdf_ingest import ingest_pdf

from agents.teacher import teacher_agent
from agents.quiz import quiz_agent
from agents.coding import coding_agent
from agents.progress import progress_agent
from agents.rag_agent import rag_agent


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="ML Academy Agent",
    page_icon="🤖",
    layout="wide"
)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📚 ML Academy")

    st.write("AI Tutor powered by Gemini + RAG")

    st.divider()

    if st.button("Show Progress"):

        progress = progress_agent()

        st.text(progress)

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        os.makedirs("data", exist_ok=True)

        pdf_path = os.path.join(
            "data",
            uploaded_file.name
        )

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        if st.button("📚 Add To Knowledge Base"):

            with st.spinner(
                "Creating chunks and embeddings..."
            ):

                chunks = ingest_pdf(pdf_path)

            st.success(
                f"Knowledge Base Updated! ({chunks} chunks)"
            )


# ==================================================
# MAIN PAGE
# ==================================================

st.title("🤖 ML Academy Agent")

st.markdown("""
### Learn:

- Machine Learning
- Deep Learning
- Python
- Data Science
- AI Engineering

Ask questions, take quizzes, upload PDFs, and learn interactively.
""")


# ==================================================
# CHAT HISTORY
# ==================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==================================================
# USER INPUT
# ==================================================

query = st.chat_input(
    "Ask me anything about Machine Learning..."
)


if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    route = route_query(query)

    with st.spinner("Thinking..."):

        try:

            if route == "quiz":

                response = quiz_agent(query)

            elif route == "coding":

                response = coding_agent(query)

            elif route == "progress":

                response = progress_agent()

            elif route == "rag":

                response = rag_agent(query)

            else:

                response = teacher_agent(query)

        except Exception as e:

            response = f"""
### Error

{str(e)}
"""

    with st.chat_message("assistant"):

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )