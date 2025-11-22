from fastapi import FastAPI, Query, Path, Body, HTTPException, UploadFile, File, Form,Header
import os
import threading
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from endpoint_documentation import document_router
from query_routing import query_router
import io
import json
import logging


# ---------------- Paths and constants ----------------
REQUEST_LOGS = "request.csv"
HISTORY_LOGS = "history.csv"  # agr available hai
PREPROCESSOR_PATH = "preprocessor.joblib"
MODEL_PATH = "iso_model.joblib"
THRESH = 100
CONTAMINATION = 0.05

categorical_features = ["endpoint", "method"]
numeric_features = ["payload_size", "response_time", "status_code"]

# ---------------- Globals ----------------
preprocessor = None
iso_model = None
model_lock = threading.Lock()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger=logging.getLogger(__name__)

# ---------------- Helper functions ----------------
def train_model_from_dataframe(df: pd.DataFrame, contamination=CONTAMINATION):
    """Train Isolation Forest model from a dataframe and save artifacts."""
    global preprocessor, iso_model
    df = df.dropna(subset=categorical_features + numeric_features)
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df.dropna(subset=numeric_features)
    if len(df) == 0:
        print("No usable rows for training.")
        return False

    # Preprocessor
    preprocessor_local = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ])
    X = preprocessor_local.fit_transform(df[categorical_features + numeric_features])

    # Isolation Forest
    iso_local = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso_local.fit(X)

    # write artifacts locally after training then push to the global path 
    tmp_pre = PREPROCESSOR_PATH + ".tmp"
    tmp_mod = MODEL_PATH + ".tmp"
    dump(preprocessor_local, tmp_pre)
    dump(iso_local, tmp_mod)
    os.replace(tmp_pre, PREPROCESSOR_PATH)
    os.replace(tmp_mod, MODEL_PATH)

    preprocessor = preprocessor_local
    iso_model = iso_local
    logger.info(f"Trained on {len(df)} rows and saved artifacts.")
    return True

def retrain_model_from_logs(df_bootstrap: pd.DataFrame = None):
    """Retrain model using logs or optional bootstrap DataFrame."""
    # If bootstrap dataframe is provided, use it for first-time training
    if df_bootstrap is not None and not df_bootstrap.empty:
        train_model_from_dataframe(df_bootstrap)
        logger.info("Trained from bootstrap DataFrame.")
        return

    # Use existing logs
    if os.path.exists(HISTORY_LOGS) and os.path.exists(REQUEST_LOGS):
        df_history = pd.read_csv(HISTORY_LOGS)
        df_requests = pd.read_csv(REQUEST_LOGS)
        df_combined = pd.concat([df_history, df_requests], ignore_index=True)
        train_model_from_dataframe(df_combined)
        df_combined.to_csv(HISTORY_LOGS, index=False)
        os.remove(REQUEST_LOGS)
        logger.info("Updated history logs after retraining.")
    elif os.path.exists(HISTORY_LOGS):
        df_history = pd.read_csv(HISTORY_LOGS)
        train_model_from_dataframe(df_history)
    elif os.path.exists(REQUEST_LOGS):
        df_requests = pd.read_csv(REQUEST_LOGS)
        train_model_from_dataframe(df_requests)
        df_requests.to_csv(HISTORY_LOGS, index=False)
    else:
        logger.warning("No data available to train.")

def maybe_trigger_retrain(df: pd.DataFrame):
    # helper to check retrain threshold
    if len(df) >= THRESH:
        retrain_model_from_logs()

def load_model_artifacts():
    global preprocessor, iso_model
    # Case 1: Model exists 
    if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
        preprocessor = load(PREPROCESSOR_PATH)
        iso_model = load(MODEL_PATH)
        logger.info("Loaded preprocessor and model.")
        return True
    else:
        logger.error("No model artifacts found.")
        return False

def predict_anomaly(df: pd.DataFrame):
    global iso_model, preprocessor
    try:
        with model_lock:
            if preprocessor is None or iso_model is None:
                raise RuntimeError("Model artifacts not loaded")

            cols = categorical_features + numeric_features
            missing = [i for i in cols if i not in df.columns]
            if missing:
                raise RuntimeError(f"Missing columns that should be there: {missing}")

            for c in numeric_features:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

            transformed = preprocessor.transform(df)
            preds = iso_model.predict(transformed)

            # Map -1 → 1 (anomaly), 1 → 0 (normal)
            return [1 if p == -1 else 0 for p in preds]

    except Exception as e:
        logger.error(f"Prediction failed due to {str(e)} ")
        return {"Status":"failed","Code":"Prediction error","Detail":str(e)}


# ---------------- FastAPI App ----------------
app = FastAPI()
app.include_router(document_router)
app.include_router(query_router)

@app.on_event("startup")
def startup_event():
    # Load model artifacts at startup if available
    load_model_artifacts()

@app.get("/")
def status():
    global preprocessor, iso_model
    return {"Status": "Success", "Code":"model loaded" ,"Detail": preprocessor is not None and iso_model is not None}

# ---------------- Validate Endpoint ----------------
class RequestData(BaseModel):
    endpoint: str
    method: str
    payload_size: int
    response_time: int
    status_code: int

@app.post('/validate/{user_id}')
async def validate_request(user_id: int = Path(..., description="The requester_id path"),
                           data: RequestData = Body(..., description="Required data for validation"),
                           verbose: Optional[bool] = Query(False, description="any verbose required")):
    df_test = pd.DataFrame([data.dict()])
    is_anomaly = predict_anomaly(df_test)
    df_test["timestamp"] = pd.Timestamp.now()
    df_test["user"] = user_id

    # Save or append to request logs atomically
    with model_lock:
        if os.path.exists(REQUEST_LOGS):
            old_df = pd.read_csv(REQUEST_LOGS)
            full_df = pd.concat([old_df, df_test], ignore_index=True)
        else:
            full_df = df_test
        tmp = REQUEST_LOGS + ".tmp"
        full_df.to_csv(tmp, index=False)
        os.replace(tmp, REQUEST_LOGS)

    maybe_trigger_retrain(full_df)

    response = {
        "user_id": user_id,
        "Status":"Success",
        "Code":"Prediciton done",
        "anomaly": is_anomaly,
        "data": data.dict()
    }
    return response

# ---------------- Train Endpoint ----------------
# ---------------------- CHANGES START HERE ----------------------
# @app.get('/train')   # <-- old endpoint (commented out)
# def train_model():
#     """
#     Developer-only endpoint to train/retrain the model.
#     Always uses logs (REQUEST_LOGS + HISTORY_LOGS). First-time bootstrap not included here,
#     can be done by calling retrain_model_from_logs(df_bootstrap) manually if needed.
#     """
#     with model_lock:
#         retrain_model_from_logs()  # always retrain from logs
#         success = load_model_artifacts()
#     return {"message": "Training/retraining done", "model_loaded": success}

@app.post('/train')  
async def train_model(X_API_KEY=Header(...),
                      file: Optional[UploadFile] = File(None),
                      data: Optional[str] = Form(None)):
    """
    Developer-only endpoint to train/retrain the model in background.
    Options:
    1. Upload a CSV file.
    2. Send a JSON array in the body (as a string field named 'data' in multipart/form-data).
    3. If neither, retrain from logs (REQUEST_LOGS + HISTORY_LOGS).

    """
    secret_key=os.getenv("TRAIN_API_KEY")
    if secret_key!=X_API_KEY:
        logger.error("Unauthorised person attempting to train the model")
        return HTTPException(status_code=402,detail="Unauthorised Access")
    df_bootstrap = None

    if file is not None:
        contents = await file.read()
        df_bootstrap = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        logger.info("Received CSV file for training.")
    elif data is not None:
        try:
            parsed = json.loads(data)
        except Exception as e:
            return {"Status":"Failed","Code":"Invalid JSON","Details":"Invalid json format"}
        if not isinstance(parsed, list):
            return {"Status":"Failed","Code":"Invalid json format","Detail":"'data' must be a JSON array of objects"}
        df_bootstrap = pd.DataFrame(parsed)
        logger.info("Received JSON payload for training.")

    def retrain_async(df):
        try:
            retrain_model_from_logs(df)
            with model_lock:
                load_model_artifacts()
            logger.info("Background retraining finished successfully.")
        except Exception as e:
            logger.error(f"Background retraining failed: {str(e)}")

    # retraining in background thread so it aint gonna block the main one 
    threading.Thread(target=retrain_async, args=(df_bootstrap,), daemon=True).start()

    return {"Status": "Success", "Code": "Training started", "Detail": "Model retraining is running in background."}

