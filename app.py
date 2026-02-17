from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import csv
from datetime import datetime


app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        features = [
            float(data["pregnancies"]),
            float(data["glucose"]),
            float(data["bloodpressure"]),
            float(data["skinthickness"]),
            float(data["insulin"]),
            float(data["bmi"]),
            float(data["dpf"]),
            float(data["age"])
        ]

        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1] * 100

        if probability < 30:
            risk_level = "Low Risk"
            color = "#22c55e"
            suggestion = "Maintain healthy diet and regular exercise."
        elif probability < 70:
            risk_level = "Moderate Risk"
            color = "#facc15"
            suggestion = "Monitor blood sugar levels and consult a doctor if needed."
        else:
            risk_level = "High Risk"
            color = "#ef4444"
            suggestion = "High probability detected. Please consult a healthcare professional immediately."

        # ✅ Save prediction to CSV (FIXED INDENTATION)
        with open("prediction_history.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now(),
                data["pregnancies"],
                data["glucose"],
                data["bloodpressure"],
                data["skinthickness"],
                data["insulin"],
                data["bmi"],
                data["dpf"],
                data["age"],
                round(probability, 2),
                risk_level
            ])

        return jsonify({
            "result": risk_level,
            "probability": round(probability, 2),
            "color": color,
            "suggestion": suggestion
        })

    except Exception as e:
        return jsonify({"error": str(e)})

import pandas as pd

@app.route("/dashboard")
def dashboard():
    try:
        df = pd.read_csv("prediction_history.csv")

        total_predictions = len(df)
        average_risk = round(df["probability"].mean(), 2)
        high_risk_count = len(df[df["risk_level"] == "High Risk"])

        records = df.to_dict(orient="records")

        return render_template("dashboard.html",
                               total=total_predictions,
                               avg=average_risk,
                               high=high_risk_count,
                               records=records)

    except:
        return "No data available yet."



if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
