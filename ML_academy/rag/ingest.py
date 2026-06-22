from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "data/ISLR.pdf"

loader = PyPDFLoader(PDF_PATH)

pages = loader.load()

print(f"Loaded {len(pages)} pages")