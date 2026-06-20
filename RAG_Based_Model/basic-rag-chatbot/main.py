
# ============================================================
# BASIC RAG CHATBOT USING:
# PDF + LangChain + Chroma + HuggingFace Embeddings + Groq
# ============================================================
#
# RAG = Retrieval Augmented Generation
#
# What this program does:
#
# 1. Load a PDF
# 2. Split PDF into smaller chunks
# 3. Convert chunks into vectors (embeddings)
# 4. Store vectors inside Chroma DB
# 5. Search relevant chunks when user asks a question
# 6. Send relevant chunks to Groq LLM
# 7. Generate answer from PDF content
#
# ============================================================
#
# COMPLETE FLOW
#
# PDF
#  ↓
# Pages
#  ↓
# Chunks
#  ↓
# Embeddings
#  ↓
# Chroma Vector DB
#  ↓
# Retriever
#  ↓
# Relevant Chunks
#  ↓
# Context
#  ↓
# Groq LLM
#  ↓
# Final Answer
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

# Loads PDF files and converts them into LangChain Documents
from langchain_community.document_loaders import PyPDFLoader

# Splits large text into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Converts text into vectors (embeddings)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma Vector Database
from langchain_chroma import Chroma

# Groq LLM
from groq import Groq

# Loads environment variables from .env file
from dotenv import load_dotenv

import os


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# Reads variables from .env file
load_dotenv()

# Read Groq API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Safety check
if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please check your .env file."
    )

# Create Groq Client
client = Groq(
    api_key=groq_api_key
)

print("Groq Client Initialized")


# ============================================================
# STEP 1: LOAD PDF
# ============================================================

print("\nLoading PDF...")

# Path to PDF
loader = PyPDFLoader("data/Pdf1.pdf")

# Read PDF pages
pages = loader.load()

print(f"Pages Loaded: {len(pages)}")

# Example:
#
# PDF
#  ↓
# Page 1
# Page 2
# Page 3
# ...


# ============================================================
# STEP 2: CHUNKING
# ============================================================

# Why Chunking?
#
# LLMs perform better on smaller pieces of text.
#
# Example:
#
# Page 1
#
# becomes:
#
# Chunk 1
# Chunk 2
# Chunk 3

print("\nCreating Chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# Convert pages into chunks
chunks = splitter.split_documents(pages)

print(f"Total Chunks: {len(chunks)}")

# Preview first chunk
print("\nFirst Chunk Preview:\n")
print(chunks[0].page_content[:500])


# ============================================================
# STEP 3: CREATE EMBEDDING MODEL
# ============================================================

# Embeddings convert text into numbers.
#
# Human:
# "Machine Learning"
#
# Computer:
# [0.12, -0.55, 0.34, ...]

print("\nLoading Embedding Model...")

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

print("Embedding Model Loaded")


# ============================================================
# STEP 4: TEST EMBEDDING
# ============================================================

# Convert first chunk into vector

vector = embedding_model.embed_query(
    chunks[0].page_content
)

print("Embedding Length:", len(vector))

print("\nFirst 10 Vector Values:\n")
print(vector[:10])

# all-MiniLM-L6-v2 produces 384 dimensions
#
# Example:
#
# Chunk
#   ↓
# [384 Numbers]


# ============================================================
# STEP 5: CREATE VECTOR DATABASE
# ============================================================

# Chroma stores embeddings.
#
# Think:
#
# Excel stores rows
#
# Chroma stores vectors

print("\nCreating Chroma Vector Database...")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("Vector DB Created")


# ============================================================
# STEP 6: CREATE RETRIEVER
# ============================================================

# Retriever = Librarian
#
# Chroma DB = Library
#
# User asks question
#       ↓
# Retriever finds best chunks

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,          # Return top 3 chunks
        "fetch_k": 10    # Look through top 10 first
    }
)

print("Retriever Created")


# ============================================================
# CHATBOT LOOP
# ============================================================

print("\n" + "=" * 60)
print("RAG CHATBOT READY")
print("Type 'exit' to quit")
print("=" * 60)

while True:

    # ========================================================
    # GET USER QUESTION
    # ========================================================

    query = input("\nAsk a question:\n> ")

    # Exit condition
    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    print(f"\nSearching for: {query}")

    # ========================================================
    # RETRIEVE RELEVANT CHUNKS
    # ========================================================

    results = retriever.invoke(query)

    print(f"\nRetrieved {len(results)} chunks")

    # Show retrieved chunks
    for i, doc in enumerate(results, start=1):

        print(f"\n----- Chunk {i} -----")

        print(doc.page_content[:300])

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    # Combine chunks into one large context
    #
    # Chunk 1
    # +
    # Chunk 2
    # +
    # Chunk 3
    # =
    # Context

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    print(f"\nContext Length: {len(context)}")

    # ========================================================
    # CREATE PROMPT
    # ========================================================

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the information provided
in the context.

If the answer cannot be found in the context,
reply with:

"I could not find the answer in the provided document."

CONTEXT:
{context}

QUESTION:
{query}
"""

    # ========================================================
    # SEND TO GROQ
    # ========================================================

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    # ========================================================
    # DISPLAY FINAL ANSWER
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(response.choices[0].message.content)

    print("\n" + "=" * 60)


# ============================================================
# END OF PROGRAM
# ============================================================

#
# Example Session
#
# Ask a question:
# > What is ensemble learning?
#
# FINAL ANSWER:
# Ensemble learning is ...
#
# Ask a question:
# > What is bagging?
#
# FINAL ANSWER:
# Bagging is ...
#
# Ask a question:
# > exit
#
# Goodbye!
#
# ============================================================
#
# FUTURE IMPROVEMENTS
#
# 1. Multiple PDFs
# 2. Streamlit UI
# 3. Source citations
# 4. Chat memory
# 5. LangGraph Agentic RAG
# 6. Persistent Chroma Loading
#
# ============================================================

