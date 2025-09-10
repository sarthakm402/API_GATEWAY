from sklearn.ensemble import IsolationForest    
import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline 
df=pd.read_csv(r"C:\Users\SarthakMohapatra\Desktop\sarthak_dev_code\misc\api_dataset.csv")
iso=IsolationForest(
    n_estimators=500,
    contamination=0.1,
    random_state=42
)
num_cols=df.select_dtypes(include=["float64","Int64","float32","Int32"]).columns.to_list()
cat_cols=df.select_dtypes(include=["object","category"]).columns.to_list()
preprocessor=ColumnTransformer(transformers=[
    ("cat",OneHotEncoder(handle_unknown="ignore"),cat_cols),
    ("num",StandardScaler(),num_cols)

])
X = preprocessor.fit_transform(df)
iso.fit(X)
preds=iso.predict(X)
print(preds)



from fastapi import FastAPI,Query,Path,Body,File
from pydantic import BaseModel
from typing import Optional, List,Dict,Any,Tuple

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
       errors["payload_size"]=" INVALID Payload size should be positive interger."
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
    
    
    
    
app=FastAPI 
@app.get("/")
def status():
    return {"status":"FAST API running."}

@app.post("/validate")
async def isvalid_or_not(item:dict):
    is_valid_rule,errors=is_valid(item)
    anomaly_flag=0
    if is_valid_rule:




        anomaly_flag=0
    final_valid = is_valid_rule and (anomaly_flag == 0)
    return {
        "valid": final_valid,
        "rule_errors": errors if not is_valid_rule else None,
        "anomaly": anomaly_flag,
        "cleaned": item if is_valid_rule else None
    }




