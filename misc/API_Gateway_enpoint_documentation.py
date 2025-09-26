from fastapi import FastAPI, Query, Path, Body
from fastapi import APIRouter
from pydantic import Basemodel,Field
import os
import pandas as pd
# help krega works as a small fastapi app that can be called by main fastapi helps to keep things clean
document_router=APIRouter(
    prefix="/docs",
    tags=["docs"]
)
REQUEST_LOGS="request.csv"
@document_router.get("/",summary="Monitored API Docs",description="Show all the enpoints with examples")
def monitor():
    if not os.path.exists(REQUEST_LOGS):
        return{"message":"No request logs found"}
    df=pd.read_csv(REQUEST_LOGS)
    if df.empty:
        return{"message":"Requests Logs although there is empty"}
    df_new=df.groupby(["endpoint","method"])
    endpoints={}
    for (end,met),i in df_new:
        example=i.iloc[0].to_dict()
        example.pop("timestamp", None)
        example.pop("user", None)
        endpoints[f'{end} {met}']=example
    return {
        "message":"Automatically generated API endpoints",
        "monitored api":endpoints,
        "note":"example for the endpoints are taken from actual calls"
    }



    
    
