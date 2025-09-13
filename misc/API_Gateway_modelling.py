from sklearn.ensemble import IsolationForest    
import pandas as pd
import numpy as np  
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline    
from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt

# ------------------ ML Model Setup ------------------
df = pd.read_csv(r"C:\Users\sarthak mohapatra\Desktop\vs ml  course code\API_GATEWAY\misc\api_dataset.csv")
iso = IsolationForest(
    n_estimators=500,
    contamination=0.1,
    random_state=42
)
num_cols = df.select_dtypes(include=["float64","Int64","float32","Int32"]).columns.to_list()
cat_cols = df.select_dtypes(include=["object","category"]).columns.to_list()
preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])
X = preprocessor.fit_transform(df)
iso.fit(X)
pred=iso.predict(X)
scores = iso.decision_function(X)  
# a little test
df_results = df.copy()
df_results["anomaly_flag"] = pred
df_results["anomaly_score"] = scores
plt.hist(scores, bins=50)
plt.title("Isolation Forest anomaly score distribution")
plt.show()
most_anomalous = df_results.sort_values(by="anomaly_score").head(10)
print(most_anomalous)


# ------------------ Validation Rules ------------------
VALID_ENDPOINTS = ["/login", "/purchase", "/get-user", "/metrics"]
VALID_METHODS = ["GET", "POST"]
VALID_STATUS_CODES = [200, 400, 401, 404, 500]

def is_valid(item:Dict[str,Any])->Tuple[bool,dict[str,str]]:
    errors={}
    required_fields = ["endpoint", "method", "payload_size", "response_time", "status_code"]
    for f in required_fields:
        if f not in item:
            errors[f]="Missing field"
    if errors:
        return False,errors
    if item["method"] not in VALID_METHODS:
        errors["method"]=f'INVALID method {item["method"]}'
    if item["endpoint"] not in VALID_ENDPOINTS:
        errors["endpoint"]=f'INVALID endpoint {item["endpoint"]}'
    if item["status_code"] not in VALID_STATUS_CODES:
        errors["status_code"]=f'INVALID status code {item["status_code"]}'
    if not (isinstance(item["payload_size"],int)) or item["payload_size"]<0:
        errors["payload_size"]="INVALID Payload size should be positive integer."
    elif item["payload_size"]>2000:
        errors["payload_size"]="INVALID Payload Size is too big."
    if item["response_time"]<0:
        errors["response_time"]="INVALID response time must be greater than 0."
    elif item["response_time"]>10000:
        errors["response_time"]="INVALID response time too long."
    if item["endpoint"] in ["/login", "/purchase"] and item["method"] != "POST":
        errors["method_endpoint"] = f"{item['endpoint']} must use POST"
    if item["endpoint"] in ["/get-user", "/metrics"] and item["method"] != "GET":
        errors["method_endpoint"] = f"{item['endpoint']} must use GET"
    if errors:
        return False ,errors
    return True ,{}

def check_model_anomaly(item:Dict[str,Any])->int:
    df = pd.DataFrame([item])
    x_item = preprocessor.transform(df)
    pred = iso.predict(x_item)[0]
    return 1 if pred == -1 else 0

# ------------------ FastAPI Setup ------------------
app = FastAPI()
class APIRequest(BaseModel):
    endpoint: str
    method: str
    payload_size: int
    response_time: int
    status_code: int

@app.get("/")
def status():
    return {"status":"FAST API running."}

@app.post("/validate/{request_id}")
async def isvalid_or_not(
    request_id: int = Path(..., description="Unique request ID"),
    item: APIRequest = Body(..., description="API request log entry"),
    verbose: Optional[bool] = Query(False, description="Verbose output flag")
):
    item_dict = item.dict()
    is_valid_rule, errors = is_valid(item_dict)
    anomaly_flag = 0
    if is_valid_rule:
        anomaly_flag = check_model_anomaly(item_dict)
    final_valid = is_valid_rule and (anomaly_flag == 0)
    return {
        "request_id": request_id,
        "valid": final_valid,
        "rule_errors": errors if not is_valid_rule else None,
        "anomaly": anomaly_flag,
        "cleaned": item_dict if is_valid_rule else None,
        "verbose": verbose
    }
