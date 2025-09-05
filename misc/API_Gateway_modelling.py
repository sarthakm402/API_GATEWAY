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
