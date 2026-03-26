import os
import time
import bcrypt
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from bson import ObjectId
from huggingface_hub import hf_hub_download

app = FastAPI(title="PhytoSphere Pro AI Engine (Cloud Edition)")

# --- CORS CONFIGURATION ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MONGODB ATLAS CONNECTION ---
DEFAULT_URI = "mongodb+srv://Tirthesh:TIRTHESH@phytosensor.ohncgpf.mongodb.net/?appName=Phytosensor"
MONGO_URI = os.getenv("MONGO_URI", DEFAULT_URI)

db = None
users_col = None
logs_col = None

try:
    client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=5000,
        tls=True,
        tlsAllowInvalidCertificates=True
    )
    client.admin.command('ping')

    db = client["phytosphere_db"]
    users_col = db["users"]
    logs_col = db["logs"]
    print("Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"CRITICAL ERROR: Could not connect to MongoDB Atlas: {e}")

# --- 🤗 HUGGING FACE MODEL LOADING ---
from huggingface_hub import hf_hub_download

model = None
MODEL_FILENAME = "phytorem_model.pkl"
REPO_ID = "sri1210/phytorem_model"

try:
    print("Downloading model from Hugging Face...")

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME
    )

    print("Model downloaded successfully.")

    model = joblib.load(model_path)
    print("Random Forest model loaded successfully.")

except Exception as e:
    print(f"Model Loading Error: {e}")

# --- HELPER FUNCTIONS ---
def format_mongo_doc(doc):
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

# --- DATA MODELS ---
class UserAuth(BaseModel):
    email: str
    password: str
    deviceId: str

class SensorPayload(BaseModel):
    cu: float
    cd: float
    pb: float
    deviceId: str

# --- AUTH ROUTES ---
@app.post("/register")
async def register(user: UserAuth):
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    if users_col.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email is already registered.")

    hashed_password = bcrypt.hashpw(user.password.encode('utf-8'), bcrypt.gensalt())

    users_col.insert_one({
        "email": user.email,
        "password": hashed_password,
        "deviceId": user.deviceId,
        "created_at": datetime.utcnow()
    })

    return {"status": "success", "message": "Registration successful."}

@app.post("/login")
async def login(user: UserAuth):
    if users_col is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    db_user = users_col.find_one({"email": user.email, "deviceId": user.deviceId})

    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid credentials or Device ID mismatch.")

    if bcrypt.checkpw(user.password.encode('utf-8'), db_user["password"]):
        return {
            "status": "success",
            "user": {
                "email": db_user["email"],
                "deviceId": db_user["deviceId"]
            }
        }

    raise HTTPException(status_code=401, detail="Incorrect password.")

# --- PREDICT ---
@app.post("/predict")
async def predict(data: SensorPayload):
    if logs_col is None:
        raise HTTPException(status_code=503, detail="Database not connected.")

    if model is None:
        raise HTTPException(status_code=500, detail="AI Model not initialized.")

    try:
        input_df = pd.DataFrame([[data.cu, data.cd, data.pb]], columns=['Copper', 'Cadmium', 'Lead'])
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_

        recommendations = sorted([
            {"name": str(classes[i]), "confidence": round(float(probs[i] * 100), 2)}
            for i in range(len(probs)) if (probs[i] * 100) > 1
        ], key=lambda x: x['confidence'], reverse=True)

        logs_col.insert_one({
            "Id": data.deviceId,
            "metals_detected": {"Lead": data.pb, "Copper": data.cu, "Cadmium": data.cd},
            "timestamp": datetime.utcnow(),
            "prediction": recommendations[0]["name"] if recommendations else "None",
            "confidence": recommendations[0]["confidence"] if recommendations else 0
        })

        return {"status": "success", "recommendations": recommendations}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- MAIN ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)