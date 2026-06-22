import os

from dotenv import load_dotenv
from google import genai

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory="vector_db",
    embedding_function=embedding_model
)


def rag_agent(question):

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    prompt = f"""
    You are a Machine Learning tutor.

    Use the provided context whenever relevant.

    If the context does not contain enough information,
    answer using your own Machine Learning knowledge.

    Context:
    {context}

    Question:
    {question}
    """

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception as e:

        return str(e)