import serial
import requests
import time
import json
import pymongo
from datetime import datetime

# ==========================================
# CONFIGURATION
# ==========================================

# 1. IDENTIFY YOUR PORT
# On macOS: ls /dev/cu.*
ARDUINO_PORT = '/dev/cu.usbmodem1101' 
BAUD_RATE = 9600

# 2. ENDPOINTS
API_URL = "http://localhost:8000/predict"

# 3. MONGODB CONFIGURATION
# Connection string for local MongoDB
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "phytosphere_db"
COLLECTION_NAME = "sensor_readings"

# ==========================================
# DATABASE LOGIC
# ==========================================

# Initialize MongoDB Client
try:
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
    db = client[DB_NAME]
    readings_col = db[COLLECTION_NAME]
    # Check connection
    client.server_info() 
    print(f"Connected to MongoDB at {MONGO_URI}")
except Exception as e:
    print(f"MongoDB Error: Could not connect. Ensure MongoDB is running. {e}")
    exit()

def log_to_mongodb(cu, cd, pb, prediction_data):
    """Stores the sensor data and ML prediction in MongoDB."""
    document = {
        "timestamp": datetime.utcnow(),
        "sensors": {
            "copper": float(cu),
            "cadmium": float(cd),
            "lead": float(pb)
        },
        "recommendation": {
            "plant": prediction_data.get('plant'),
            "common_name": prediction_data.get('common_name'),
            "confidence": prediction_data.get('confidence'),
            "reliability": prediction_data.get('reliability')
        }
    }
    try:
        readings_col.insert_one(document)
    except Exception as e:
        print(f"Error saving to MongoDB: {e}")

# ==========================================
# BRIDGE LOGIC
# ==========================================

def start_bridge():
    print("--- PhytoSphere Pro: Arduino to MongoDB Bridge ---")
    
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        ser.flush()
        print(f"Successfully connected to Arduino on {ARDUINO_PORT}")
    except Exception as e:
        print(f"Serial Error: {e}")
        return

    print(f"Logging data to MongoDB Database: {DB_NAME}")

    while True:
        if ser.in_waiting > 0:
            try:
                raw_line = ser.readline().decode('utf-8').strip()
                if raw_line:
                    parts = raw_line.split(',')
                    if len(parts) == 3:
                        cu, cd, pb = parts
                        
                        # 1. Get Prediction from ML Model
                        payload = {"cu": float(cu), "cd": float(cd), "pb": float(pb)}
                        response = requests.post(API_URL, json=payload)
                        
                        if response.status_code == 200:
                            prediction_data = response.json()
                            
                            # 2. Store everything in MongoDB
                            log_to_mongodb(cu, cd, pb, prediction_data)
                            
                            print(f"DB SAVED: Cu:{cu} Pb:{pb} -> Recommendation: {prediction_data['plant']}")
                        else:
                            print(f"Backend Error: {response.status_code}")
                            
            except Exception as e:
                print(f"Error: {e}")
        
        time.sleep(0.1)

if __name__ == "__main__":
    start_bridge()