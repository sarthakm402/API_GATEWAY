from fastapi import APIRouter, Body
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.vectorstores import Chroma
import pandas as pd
import os
import logging
GOOGLE_API_KEY = "AIzaSyBDopiFyq_IpE6WT3vaHoV6cV8pByUUHIg"

query_router = APIRouter(
    prefix="/query",
    tags=["query"]
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger=logging.getlogger(__name__)
QUERY_LOGS = "request.csv"

def build_vectorstore(csv_file=QUERY_LOGS):
    if not os.path.exists(csv_file):
        logger.warning(f"Query logs file {csv_file} not found.")
        return None

    df = pd.read_csv(csv_file)
    if df.empty:
        logger.warning(f"Query logs file {csv_file} is empty.")
        return None

    documents = [
        Document(
            page_content=str(row.to_dict()),
            metadata={"timestamp": row.get("timestamp"), "user_id": row.get("user_id")}
        )
        for _, row in df.iterrows()
    ]
   
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 

    vectorstore = Chroma.from_documents(documents, embeddings)
    logger.info(f"Vectorstore built with {len(documents)} documents.")
    return vectorstore

vectorstore = build_vectorstore()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    api_key=GOOGLE_API_KEY
)

if vectorstore:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
else:
    qa_chain = None

@query_router.post("/", summary="Query the logs in natural English")
async def query(question: str = Body(..., description="Ask a question about the logs")):
    if not qa_chain:
        logger.warning("Query attempted but no vectorstore is available.")
        return {"Status":"Fail","Code":"no logs","Detail": "No logs available to query."}

    try:
        answer = qa_chain.run(question)
        logger.info(f"Query executed successfully: {question}")
        return {"Status":"Success","Detail":{"question":question, "answer": answer}}
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        return {"Status":"Fail","Code":"Error","Detail": str(e)}
