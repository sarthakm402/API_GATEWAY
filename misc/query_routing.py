from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pandas as pd
from fastapi import APIRouter, Body
import os

query_router = APIRouter(
    prefix="/query",
    tags=["query"]
)

QUERY_LOGS = "request.csv"
def build_vectorstore(csv_file=QUERY_LOGS):
    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)
    if df.empty:
        return None

    documents = [
        Document(
            page_content=str(row.to_dict()),
            metadata={"timestamp": row.get("timestamp"), "user_id": row.get("user_id")}
        )
        for _, row in df.iterrows()
    ]

    embeddings = GoogleGenerativeAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore

vectorstore = build_vectorstore()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",api_key="AIzaSyBDopiFyq_IpE6WT3vaHoV6cV8pByUUHIg")
if vectorstore:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=vectorstore.as_retriever()
    )
else:
    qa_chain = None

@query_router.post("/", summary="Query on Logs with normal English")
async def query(question: str = Body(..., description="Ask a query in English")):
    if not qa_chain:
        return {"message": "No logs available to query."}

    try:
        answer = qa_chain.run(question)
        return {"question": question, "answer": answer}
    except Exception as e:
        return {"error": str(e)}
