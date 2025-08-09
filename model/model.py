from flask import Flask, request, jsonify
import pickle
import numpy as np
import requests
import pandas as pd
import random
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Load model once at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("knn_model.pkl", "rb") as f:
    knn = pickle.load(f)

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
    return knn.predict([[lat, lon]])[0]

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

    if weather is None:
        return jsonify({"error": "Weather data unavailable"}), 503

    X = np.array([[weather["temperature"], weather["relative_humidity"], weather["wind_speed"], vegetation_type]])
    prob = model.predict_proba(X)[0][1]

    return jsonify({
        "probability": round(prob, 3),
        "weather": weather
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
