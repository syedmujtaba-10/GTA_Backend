import os
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain


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
# rag_chain = RetrievalQA.from_chain_type(
#     llm=ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY),
#     retriever=retriever
# )

custom_prompt = PromptTemplate(
    template="""
You are a helpful government policy assistant. Use the following context to answer the user's question. Suggest only two most relevant forms

Always return relevant government policies in JSON format. 
For each policy, include:
- policy_name
- description
- Form in JSON 
- form_link 

Context:
{context}

Question: {question}
Answer:
""",
    input_variables=["context", "question"]
)

# Load QA chain with prompt
qa_chain = load_qa_chain(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY),
    chain_type="stuff",
    prompt=custom_prompt
)

# Use the chain in RetrievalQA
rag_chain = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)


@app.get("/query")
async def query_government_policy(question: str = Query(..., description="Ask a question about government policies or programs. Always return policies with Form in JSON")):
    """
    Passes the question to the RAG agent for a detailed explanation based on government policy summaries.
    """
    response = rag_chain.run(question)
    return {
        "question": question,
        "answer": response
    }
