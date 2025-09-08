import pandas as pd
import numpy as np 
np.random.seed(32) 
n=50000    
df=pd.DataFrame({
"endpoint":np.random.choice(["/login","/purchase","/get-user","/matrics"],n),
"method":np.random.choice(["GET","POST"],n),
"payload_size":np.random.randint(50,2000,n),
"response_time":np.random.randint(n)*2,
"status_code":np.random.choice([200,400,401,404,500],n,p=[0.9,0.02,0.02,0.03,0.03])
}) 
index=np.random.choice(df.index,1000,replace=False)
df.loc[index,"payload_size"]*=20
df.loc[index,"response_time"]*=10
df.to_csv("api_dataset.csv",index=False)
