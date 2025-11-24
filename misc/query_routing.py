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
logger=logging.getLogger(__name__)
QUERY_LOGS = "request.csv"

def build_vectorstore(csv_file=QUERY_LOGS):
    if not os.path.exists(csv_file):
        logger.warning(f"No csv file found at {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    if df.empty:
        logger.warning("The csv file is empty")
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
    logger.info("Vectorstore has been created")
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
    logger.error("No vectorstore present")
    qa_chain = None

@query_router.post("/", summary="Query the logs in natural English")
async def query(question: str = Body(..., description="Ask a question about the logs")):
    if not qa_chain:
        logger.error("No QA retrieval chain found")
        return {"Status":"Failed","Code":"Invalid Logs","Detail": "No logs available to query."}

    try:
        answer = qa_chain.run(question)
        logger.info("qa chain successful")
        return {"Status": "Success","Detail":{"question": question, "answer": answer}}
    except Exception as e:
        logger.warning(f"Error occured {str(e)}")
        return {"Status":"Failed","Code":"Error","Detail": str(e)}
