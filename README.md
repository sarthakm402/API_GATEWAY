# FastAPI Model Gateway

This project implements an **API Gateway** using **FastAPI** that routes incoming requests to different ML models or processing endpoints.  
It also includes query routing, preprocessing, and model inference logic for modular and scalable design.

---
 
## 🚀 Features

- **FastAPI-powered REST API**
- **Query Routing Layer** for directing requests to appropriate models
- **IsolationForest-based Anomaly Detection**
- **Joblib-based Model Serialization**
- **Dynamic Column Transformation** using `ColumnTransformer`
- **Pydantic Models** for request validation
- **Multithreaded Request Handling**
- **Endpoints with Form, File, and JSON support**
- **Modular structure** with `document_router` for endpoint documentation

---

## 🧩 Project Structure

```
API_GATEWAY/
│
├── main.py                  # Entry point (FastAPI app)
├── query_routing.py         # Query routing logic
├── enpoint_documentation.py # Endpoint descriptions
├── model.joblib             # Serialized ML model
├── requirements.txt         # Dependencies
└── README.md                # Project documentation (this file)
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/sarthakm402/API_GATEWAY.git
cd api-gateway
```

### 2️⃣ Create and Activate a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Setup

The project uses a serialized **Isolation Forest** model trained for anomaly detection.  
You can replace `model.joblib` with any scikit-learn compatible model.

To retrain and save a model:
```python
from sklearn.ensemble import IsolationForest
from joblib import dump

model = IsolationForest()
model.fit(X_train)
dump(model, "model.joblib")
```

---

## 🚪 Running the Server

To start the FastAPI app:
```bash
uvicorn main:app --reload
```

Server runs on:
```
http://127.0.0.1:8000
```

You can also access:
```
Swagger UI: http://127.0.0.1:8000/docs
```

---

## 📡 Example Endpoints

### ➤ `/predict` — Model Inference
Accepts JSON, CSV, or form data.  
Runs preprocessing, routes query, and returns prediction.

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "feature1": 3.4,
  "feature2": "A",
  "feature3": 12
}'
```

**Example Response:**
```json
{
  "prediction": "normal"
}
```

### ➤ `/upload` — Upload File Endpoint
Uploads CSV files for batch prediction.

### ➤ `/docs` — API Documentation
Interactive documentation powered by Swagger UI.

---

## 🧠 Query Routing

The `query_routing.py` file defines rules to direct incoming data to the appropriate model or processing function.  
For instance:
```python
if "anomaly" in query_type:
    route_to(isolation_forest_model)
else:
    route_to(default_model)
```

---

## 🧰 Environment Variables

If you want to use environment-specific configurations, create a `.env` file:
```
MODEL_PATH=model.joblib
PORT=8000
LOG_LEVEL=info
```

---

## 🧪 Testing

Run tests using:
```bash
pytest
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Sarthak Mohapatra**  
💼 AI Developer | ⚙️ Machine Learning | 🧠 FastAPI & Model Deployment
