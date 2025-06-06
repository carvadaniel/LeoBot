# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS setup so your HTML/JS frontend can talk to the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace * with specific origin like "http://127.0.0.1:5500"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

# Load once at startup
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/", embed_model)
retriever = db.as_retriever(search_kwargs={"k": 10})
llm = Ollama(model="llama3")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    response = qa.run(request.query)
    return {"response": response}
