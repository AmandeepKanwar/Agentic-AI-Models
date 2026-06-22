from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("data/islr.pdf")

pages = loader.load()

print(f"Pages Loaded: {len(pages)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(pages)

print(f"Chunks Created: {len(chunks)}")

print("\nFirst Chunk Preview:\n")
print(chunks[0].page_content[:500])