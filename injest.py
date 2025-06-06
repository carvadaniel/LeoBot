from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

DOCS_DIR = "docs"
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
documents = []
for fname in os.listdir(DOCS_DIR):
    if fname.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(os.path.join(DOCS_DIR, fname))
        documents.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(documents)
db = FAISS.from_documents(docs, embed_model)
db.save_local("vectorstore")
print("Vectorstore built!")