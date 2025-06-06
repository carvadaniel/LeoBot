from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import os
 
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for development; restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
class Query(BaseModel):
    query: str
 
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/", embed_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 10})
llm = Ollama(model="llama3")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
 
DOCS_DIR = "docs"  # Adjust if your docs folder is elsewhere

@app.post("/chat")
async def chat(q: Query):
    user_query = q.query.lower().strip()
    if user_query in ["hello", "hi", "hey"]:
        return {"answer": "Hello! How can I help you with your documents?"}
    if "how many documents" in user_query:
        num_docs = len([f for f in os.listdir(DOCS_DIR) if f.endswith(".docx")])
        return {"answer": f"You have {num_docs} documents in your docs folder."}
    docs = retriever.get_relevant_documents(q.query)
    if not docs:
        # Fallback: let the LLM answer from its general knowledge
        answer = llm(q.query)
        return {"answer": answer}
    answer = qa.run(q.query)
    return {"answer": answer}