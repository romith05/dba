from flask import Flask, request, jsonify
import pickle
import numpy as np
import requests
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Load model once at startup
with open("/data/rf_model.pkl", "rb") as rf:
    rfm = pickle.load(rf)
with open("/data/my_forest_encoder.pkl","rb") as rfle:
    rfm_label_encoder = pickle.load(rfle)
with open("/data/knn_model.pkl", "rb") as knn:
    knnm = pickle.load(knn)
with open("/data/knn_encoder.pkl", "rb") as knnle:
    knn_label_encoder = pickle.load(knnle)

executor = ThreadPoolExecutor(max_workers=2)

def fetch_weather(lat, lon):
    date = pd.Timestamp.now().strftime('%Y-%m-%d')
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,windspeed_10m"
        f"&timezone=MST"
    )
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        hourly = data.get("hourly", {})
        hours_available = range(len(hourly.get("temperature_2m", [])))

        if not hours_available:
            return None

        random_hour = random.choice(hours_available)
        return {
            "temperature": hourly.get("temperature_2m", [None]*24)[random_hour],
            "relative_humidity": hourly.get("relative_humidity_2m", [None]*24)[random_hour],
            "wind_speed": hourly.get("windspeed_10m", [None]*24)[random_hour],
        }
    except Exception as e:
        print(f"⚠️ Weather fetch error: {e}")
        return None

def knn_predict(lat, lon):
    return knnm.predict([[lat, lon]])[0]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    lat = data.get("lat")
    lon = data.get("lon")

    if lat is None or lon is None:
        return jsonify({"error": "Missing latitude or longitude"}), 400

    # Launch both tasks in parallel
    future_weather = executor.submit(fetch_weather, lat, lon)
    future_vegtype = executor.submit(knn_predict, lat, lon)

    # Wait for results
    weather = future_weather.result()
    vegetation_type = future_vegtype.result()
    vegetation_label = knn_label_encoder.inverse_transform([vegetation_type])[0]

    if weather is None:
        #return jsonify({"error": "Weather data unavailable"}), 503
        weather = { "temperature": 10, "relative_humidity": 0, "wind_speed": 100}
    
    # Prepare features for prediction   
    if vegetation_type is None:
        return jsonify({"error": "Vegetation type prediction failed"}), 503
    X = np.array([[weather["temperature"], weather["relative_humidity"], weather["wind_speed"], rfm_label_encoder.transform([vegetation_label])[0]]])

    prob = rfm.predict_proba(X)[0][1]

    return jsonify({
        "probability": round(prob, 3),
        "weather": weather,
        "vegetation_type": vegetation_label
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
