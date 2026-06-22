from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load PDF
loader = PyPDFLoader("data/islr.pdf")
pages = loader.load()

# Chunk PDF
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(pages)

print(f"Chunks Created: {len(chunks)}")


# Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store in ChromaDB
db = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="vector_db"
)

print("Vector Database Created!")