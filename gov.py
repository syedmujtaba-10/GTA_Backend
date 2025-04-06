import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️ Please set the OPENAI_API_KEY in your environment variables.")

# Load vector store built on government policy summaries
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = FAISS.load_local("faiss_atkins_db", embeddings=embedding_model, allow_dangerous_deserialization=True)
retriever = vector_store.as_retriever()

# Define the Retrieval-Augmented Generation (RAG) chain
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY),
    retriever=retriever
)

@app.get("/query")
async def query_government_policy(question: str = Query(..., description="Ask a question about government policies or programs")):
    """
    Passes the question to the RAG agent for a detailed explanation based on government policy summaries.
    """
    response = rag_chain.run(question)
    return {
        "question": question,
        "answer": response
    }
