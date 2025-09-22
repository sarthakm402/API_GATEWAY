# from sklearn.ensemble import IsolationForest    
# import pandas as pd
# import numpy as np  
# from sklearn.compose import ColumnTransformer 
# from sklearn.preprocessing import OneHotEncoder, StandardScaler 
# from sklearn.pipeline import Pipeline    
# from fastapi import FastAPI, Query, Path, Body
# from pydantic import BaseModel
# from typing import Dict, Any, Tuple, Optional
# import matplotlib.pyplot as plt

# # ------------------ ML Model Setup ------------------
# df = pd.read_csv(r"C:\Users\sarthak mohapatra\Desktop\vs ml  course code\API_GATEWAY\misc\api_dataset.csv")
# iso = IsolationForest(
#     n_estimators=500,
#     contamination=0.1,
#     random_state=42
# )
# num_cols = df.select_dtypes(include=["float64","Int64","float32","Int32"]).columns.to_list()
# cat_cols = df.select_dtypes(include=["object","category"]).columns.to_list()
# preprocessor = ColumnTransformer(transformers=[
#     ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
#     ("num", StandardScaler(), num_cols)
# ])
# X = preprocessor.fit_transform(df)
# iso.fit(X)
# pred=iso.predict(X)
# scores = iso.decision_function(X)  
# # a little test
# df_results = df.copy()
# df_results["anomaly_flag"] = pred
# df_results["anomaly_score"] = scores
# plt.hist(scores, bins=50)
# plt.title("Isolation Forest anomaly score distribution")
# plt.show()
# most_anomalous = df_results.sort_values(by="anomaly_score").head(10)
# print(most_anomalous)


# # ------------------ Validation Rules ------------------
# VALID_ENDPOINTS = ["/login", "/purchase", "/get-user", "/metrics"]
# VALID_METHODS = ["GET", "POST"]
# VALID_STATUS_CODES = [200, 400, 401, 404, 500]

# def is_valid(item:Dict[str,Any])->Tuple[bool,dict[str,str]]:
#     errors={}
#     required_fields = ["endpoint", "method", "payload_size", "response_time", "status_code"]
#     for f in required_fields:
#         if f not in item:
#             errors[f]="Missing field"
#     if errors:
#         return False,errors
#     if item["method"] not in VALID_METHODS:
#         errors["method"]=f'INVALID method {item["method"]}'
#     if item["endpoint"] not in VALID_ENDPOINTS:
#         errors["endpoint"]=f'INVALID endpoint {item["endpoint"]}'
#     if item["status_code"] not in VALID_STATUS_CODES:
#         errors["status_code"]=f'INVALID status code {item["status_code"]}'
#     if not (isinstance(item["payload_size"],int)) or item["payload_size"]<0:
#         errors["payload_size"]="INVALID Payload size should be positive integer."
#     elif item["payload_size"]>2000:
#         errors["payload_size"]="INVALID Payload Size is too big."
#     if item["response_time"]<0:
#         errors["response_time"]="INVALID response time must be greater than 0."
#     elif item["response_time"]>10000:
#         errors["response_time"]="INVALID response time too long."
#     if item["endpoint"] in ["/login", "/purchase"] and item["method"] != "POST":
#         errors["method_endpoint"] = f"{item['endpoint']} must use POST"
#     if item["endpoint"] in ["/get-user", "/metrics"] and item["method"] != "GET":
#         errors["method_endpoint"] = f"{item['endpoint']} must use GET"
#     if errors:
#         return False ,errors
#     return True ,{}

# def check_model_anomaly(item:Dict[str,Any])->int:
#     df = pd.DataFrame([item])
#     x_item = preprocessor.transform(df)
#     pred = iso.predict(x_item)[0]
#     return 1 if pred == -1 else 0

# # ------------------ FastAPI Setup ------------------
# app = FastAPI()
# class APIRequest(BaseModel):
#     endpoint: str
#     method: str
#     payload_size: int
#     response_time: int
#     status_code: int

# @app.get("/")
# def status():
#     return {"status":"FAST API running."}

# @app.post("/validate/{request_id}")
# async def isvalid_or_not(
#     request_id: int = Path(..., description="Unique request ID"),
#     item: APIRequest = Body(..., description="API request log entry"),
#     verbose: Optional[bool] = Query(False, description="Verbose output flag")
# ):
#     item_dict = item.dict()
#     is_valid_rule, errors = is_valid(item_dict)
#     anomaly_flag = 0
#     if is_valid_rule:
#         anomaly_flag = check_model_anomaly(item_dict)
#     final_valid = is_valid_rule and (anomaly_flag == 0)
#     return {
#         "request_id": request_id,
#         "valid": final_valid,
#         "rule_errors": errors if not is_valid_rule else None,
#         "anomaly": anomaly_flag,
#         "cleaned": item_dict if is_valid_rule else None,
#         "verbose": verbose
#     }




###### MORE GENERALISED API ANOMALY DETECTION########
from fastapi import FastAPI, Request
import os
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

REQUEST_LOGS = "request.csv"
HISTORY_LOGS = "history.csv"#if available
PREPROCESSOR_PATH = "preprocessor.joblib"
MODEL_PATH = "iso_model.joblib"
THRESH = 100
CONTAMINATION = 0.05

categorical_features = ["endpoint", "method"]
numeric_features = ["payload_size", "response_time", "status_code"]

preprocessor = None
iso_model = None

def train_from_df(df, contamination=CONTAMINATION):
    global preprocessor, iso_model
    df = df.dropna(subset=categorical_features + numeric_features)
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df = df.dropna(subset=numeric_features)
    if len(df) == 0:
        print("No usable rows for training.")
        return False
    preprocessor_local = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        ("num", StandardScaler(), numeric_features)
    ])
    X = preprocessor_local.fit_transform(df[categorical_features + numeric_features])
    iso_local = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso_local.fit(X)
    dump(preprocessor_local, PREPROCESSOR_PATH)
    dump(iso_local, MODEL_PATH)
    preprocessor = preprocessor_local
    iso_model = iso_local
    print(f"Trained on {len(df)} rows and saved artifacts.")
    return True

def retrain():
    if os.path.exists(HISTORY_LOGS) and os.path.exists(REQUEST_LOGS):
        df_history = pd.read_csv(HISTORY_LOGS)
        df_requests = pd.read_csv(REQUEST_LOGS)
        df_combined = pd.concat([df_history, df_requests], ignore_index=True)
        train_from_df(df_combined)
        df_combined.to_csv(HISTORY_LOGS, index=False)
        print("Updated history logs after retraining.")
    elif os.path.exists(HISTORY_LOGS):
        df_history = pd.read_csv(HISTORY_LOGS)
        train_from_df(df_history)
    elif os.path.exists(REQUEST_LOGS):
        df_requests = pd.read_csv(REQUEST_LOGS)
        if len(df_requests) >= THRESH:
            train_from_df(df_requests)
    else:
        print("No data available to train.")

def load_artifacts():
    global preprocessor, iso_model
    if os.path.exists(PREPROCESSOR_PATH) and os.path.exists(MODEL_PATH):
        preprocessor = load(PREPROCESSOR_PATH)
        iso_model = load(MODEL_PATH)
        print("Loaded preprocessor and model.")
    else:
        retrain()
