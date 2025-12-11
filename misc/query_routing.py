from fastapi import APIRouter, Body
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import pandas as pd
import os
import logging
import time
import threading
import shutil
from typing import Optional, List


 
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyBDopiFyq_IpE6WT3vaHoV6cV8pByUUHIg")
QUERY_LOGS = "request.csv"
VECTORSTORE_DIR = "vectorstore_db"

DEFAULT_MODEL = os.environ.get("GOOGLE_GENAI_MODEL", "models/text-bison-001")

query_router = APIRouter(
    prefix="/query",
    tags=["query"]
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def make_llm(model_name: str | None = None):
    m = model_name or DEFAULT_MODEL
    return ChatGoogleGenerativeAI(model=m, api_key=GOOGLE_API_KEY)

llm = make_llm(os.environ.get("GOOGLE_GENAI_PRIMARY", "gemini-1.5-pro"))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore: Optional[Chroma] = None
vec_lock = threading.Lock()
last_loaded_time: Optional[float] = None


 

def build_vectorstore(csv_file=QUERY_LOGS) -> Optional[Chroma]:
    """Build Chroma DB from request.csv logs."""
    if not os.path.exists(csv_file):
        logger.warning(f"No CSV found: {csv_file}")
        return None

    df = pd.read_csv(csv_file)
    if df.empty:
        logger.warning("CSV file is empty.")
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

    with vec_lock:
        vs = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=VECTORSTORE_DIR
        )
        try:
            vs.persist()
        except Exception:
            pass

    logger.info("Vectorstore created successfully.")
    return vs


def load_vectorstore() -> Optional[Chroma]:
    """Load existing vectorstore."""
    if os.path.exists(VECTORSTORE_DIR):
        try:
            return Chroma(
                persist_directory=VECTORSTORE_DIR,
                embedding_function=embeddings
            )
        except Exception as e:
            logger.warning(f"Failed to load vectorstore: {e}")
    return None


def ensure_vectorstore_loaded():
    """Lazy load vectorstore when first query is made."""
    global vectorstore, last_loaded_time

    if vectorstore is not None:
        return

    vs = load_vectorstore()
    if not vs:
        vs = build_vectorstore()
    if not vs:
        logger.warning("Vectorstore build failed.")
        return

    vectorstore = vs
    last_loaded_time = time.time()
    logger.info("Vectorstore loaded and ready.")


 

def run_query(question: str, top_k: int = 4) -> str:
    """Run a natural-language query over log vectorstore."""
    ensure_vectorstore_loaded()

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
        except Exception as e:
            logger.warning(f"Primary model failed: {e}")
            
            try:
                fallback_llm = make_llm()
                logger.info(f"Retrying with fallback model: {DEFAULT_MODEL}")
                res = fallback_llm.invoke(messages)
                return getattr(res, "content", str(res))
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                raise

    
    return str(llm(messages))


 

@query_router.post("/", summary="Query logs in natural English")
async def query_endpoint(question: str = Body(..., example="Which endpoint had the longest response time?")):
    """
    Usage:
    POST /query/
    Body (raw string):
        "Which endpoint was slow?"
    """
    try:
        answer = run_query(question)
        return {
            "Status": "Success",
            "Detail": {
                "question": question,
                "answer": answer
            }
        }
    except Exception as e:
        logger.warning(f"Error: {e}")
        return {"Status": "Failed", "Detail": str(e)}


@query_router.post("/refresh", summary="Emergency refresh of vectorstore")
async def manual_refresh():
    """Delete old DB and rebuild."""
    global vectorstore, last_loaded_time

    with vec_lock:
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR)

        vectorstore = build_vectorstore()

    if not vectorstore:
        return {"Status": "Failed", "Detail": "Build failed"}

    last_loaded_time = time.time()
    return {"Status": "Success", "Detail": "Vectorstore refreshed!"}
