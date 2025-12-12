import os
import time
import shutil
import threading
import asyncio
import pandas as pd
from typing import Optional, List
from fastapi import APIRouter, Body
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyAavWKs0rwaaMap84Di88zrLbnWdygwwqY")
QUERY_LOGS = "request.csv"
VECTORSTORE_DIR = "vectorstore_db"
DEFAULT_MODEL = os.environ.get("GOOGLE_GENAI_MODEL", "gemini-2.5-flash")

query_router = APIRouter(prefix="/query", tags=["query"])

llm = ChatGoogleGenerativeAI(model=os.environ.get("GOOGLE_GENAI_PRIMARY", "gemini-2.5-flash"), api_key=GOOGLE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore: Optional[Chroma] = None
async_lock = asyncio.Lock()
last_loaded_time: Optional[float] = None

def build_vectorstore(csv_file=QUERY_LOGS) -> Optional[Chroma]:
    if not os.path.exists(csv_file):
        return None
    df = pd.read_csv(csv_file)
    if df.empty:
        return None
    documents: List[Document] = []
    for _, row in df.iterrows():
        content = (
            f"API endpoint {row.get('endpoint', '<unknown>')} was called using {row.get('method', '<unknown>')} method. "
            f"Payload size: {row.get('payload_size', '?')} KB. "
            f"Response time: {row.get('response_time', '?')} ms. "
            f"Status code: {row.get('status_code', '?')}."
        )
        documents.append(Document(page_content=content, metadata={"timestamp": row.get("timestamp")}))
    vs = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=VECTORSTORE_DIR)
    try:
        vs.persist()
    except Exception:
        pass
    return vs

def load_vectorstore() -> Optional[Chroma]:
    if os.path.exists(VECTORSTORE_DIR):
        try:
            return Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
        except Exception:
            return None
    return None

async def ensure_vectorstore_loaded():
    global vectorstore, last_loaded_time
    async with async_lock:
        if vectorstore is not None:
            return
        loop = asyncio.get_running_loop()
        vs = await loop.run_in_executor(None, load_vectorstore)
        if not vs:
            vs = await loop.run_in_executor(None, build_vectorstore)
        vectorstore = vs
        last_loaded_time = time.time()

async def run_query(question: str, top_k: int = 4) -> str:
    await ensure_vectorstore_loaded()
    if vectorstore is None:
        raise RuntimeError("Vectorstore unavailable")
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    try:
        docs = retriever.invoke(question)
    except Exception as e:
        raise RuntimeError(f"Retriever failed: {e}")
    if not docs:
        return "No relevant logs found."
    context = "\n\n".join([d.page_content for d in docs])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You answer ONLY from the context. If not in context, say 'I don't know.'"),
        ("user", "Context:\n{context}\n\nQuestion:\n{input}\n\nAnswer:")
    ])
    formatted = prompt.format_prompt(context=context, input=question)
    try:
        messages = formatted.to_messages()
    except:
        messages = str(formatted)
    if hasattr(llm, "invoke"):
        try:
            res = llm.invoke(messages)
            return getattr(res, "content", str(res))
        except:
            fallback_llm = ChatGoogleGenerativeAI(model=DEFAULT_MODEL, api_key=GOOGLE_API_KEY)
            res = fallback_llm.invoke(messages)
            return getattr(res, "content", str(res))
    return str(llm(messages))

@query_router.post("/", summary="Query logs in natural English")
async def query_endpoint(question: str = Body(..., example="Which endpoint had the longest response time?")):
    answer = await run_query(question)
    return {"Status": "Success", "Detail": {"question": question, "answer": answer}}

@query_router.post("/refresh", summary="Emergency refresh of vectorstore")
async def manual_refresh():
    global vectorstore, last_loaded_time
    async with async_lock:
        loop = asyncio.get_running_loop()
        def refresh_blocking():
            if os.path.exists(VECTORSTORE_DIR):
                shutil.rmtree(VECTORSTORE_DIR)
            return build_vectorstore()
        vectorstore_result = await loop.run_in_executor(None, refresh_blocking)
        if not vectorstore_result:
            return {"Status": "Failed", "Detail": "Build failed"}
        vectorstore = vectorstore_result
        last_loaded_time = time.time()
    return {"Status": "Success", "Detail": "Vectorstore refreshed!"}
