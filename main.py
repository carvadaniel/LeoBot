from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
app.mount("/static", StaticFiles(directory="static"), name="static")

class Query(BaseModel):
    query: str
 
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.load_local("vectorstore/", embed_model, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 15})
llm_ollama = Ollama(model="llama3")
llm_mistral = Ollama(model="mistral")
qa = RetrievalQA.from_chain_type(llm=llm_ollama, retriever=retriever)
 
DOCS_DIR = "docs"  # Adjust if your docs folder is elsewhere
IMG_DIR = "static/"  # Directory where images are stored

@app.post("/chat")
async def chat(q: Query):
    user_query = q.query.lower().strip()
    # List of greetings and general questions
    general_questions = [
        "hello", "hi", "hey", "greetings", "howdy", "what's up", "sup", "hi there", "how are you"
    ]
    clarification_phrases = [
        "yes", "no", "maybe", "correct", "you are correct", "that's right",
        "please clarify", "can you clarify", "what do you mean", "i don't understand",
        "could you explain", "sure", "okay", "ok", "thanks", "thank you"
    ]
    if user_query in general_questions or user_query in clarification_phrases:
        prompt = (
            "You are a helpful assistant. Answer clearly and concisely.\n"
            f"User: {q.query}\nAssistant:"
        )
        answer = llm_mistral(prompt)
        return {"answer": answer}
    if "how many documents" in user_query:
        num_docs = len([f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".docx")])
        return {"answer": f"You have {num_docs} documents in your docs folder."}
    docs = retriever.get_relevant_documents(q.query)
    # If no relevant docs, use Mistral for general knowledge
    if not docs or not docs[0].page_content.strip():
        answer = llm_mistral(q.query)
        return {"answer": answer}
    # Otherwise, use Ollama (llama3) with a custom prompt and show the source
    context = "\n".join([doc.page_content for doc in docs[:3]])
    prompt = (
        "You are a helpful assistant for document Q&A. "
        "If the user asks a general question, answer concisely. "
        "If the user asks about a document, cite the source at the end. "
        "If the question is unclear, politely ask for clarification."
        f"\n\nContext:\n{context}\n\nUser: {q.query}\nAssistant:"
    )
    answer = llm_ollama(prompt)
    top_source = docs[0].metadata.get("source", "Unknown")
    answer += f"<br><br><b>Source:</b> {top_source}"
    return {"answer": answer}