from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import pickle
import csv
import os
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import send_file
import io

app = Flask(__name__)
app.secret_key = "secret123"

# Load model
model = pickle.load(open("model.pkl", "rb"))


# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "admin" and password == "1234":
            session["user"] = username
            return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# ================= HOME =================
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("index.html")


# ================= PREDICT =================
@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

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

        # Create CSV if not exists
        file_exists = os.path.isfile("prediction_history.csv")

        with open("prediction_history.csv", mode="a", newline="") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "timestamp", "name", "patient_id", "pregnancies", "glucose", "bloodpressure",
                    "skinthickness", "insulin", "bmi", "dpf", "age",
                    "probability", "risk_level"
                ])

            writer.writerow([
                datetime.now(),
                data["name"],
                data["patient_id"],
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


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        if not os.path.exists("prediction_history.csv"):
            return render_template("dashboard.html", total=0, avg=0, high=0, records=[])

        df = pd.read_csv("prediction_history.csv")
        df.columns = df.columns.str.strip()

        total_predictions = len(df)
        average_risk = round(df["probability"].mean(), 2)
        high_risk_count = len(df[df["risk_level"] == "High Risk"])

        records = df.to_dict(orient="records")

        return render_template(
            "dashboard.html",
            total=total_predictions,
            avg=average_risk,
            high=high_risk_count,
            records=records
        )

    except Exception as e:
        return f"Dashboard Error: {e}"

@app.route("/profile/<int:index>")
def profile(index):
    if "user" not in session:
        return redirect(url_for("login"))

    df = pd.read_csv("prediction_history.csv")

    if index >= len(df):
        return "Record not found"

    record = df.iloc[index].to_dict()

    return render_template("profile.html", r=record, id=index)

@app.route("/download/<int:id>")
def download_pdf(id):
    if "user" not in session:
        return redirect(url_for("login"))

    try:
        df = pd.read_csv("prediction_history.csv")

        if id >= len(df):
            return "Record not found"

        record = df.iloc[id]

        file_name = "report.pdf"
        c = canvas.Canvas(file_name, pagesize=letter)

        c.setFont("Helvetica", 12)

        c.drawString(100, 750, "Diabetes Risk Report")
        c.drawString(100, 720, f"Name: {record['name']}")
        c.drawString(100, 700, f"Patient ID: {record['patient_id']}")
        c.drawString(100, 680, f"Glucose: {record['glucose']}")
        c.drawString(100, 660, f"BMI: {record['bmi']}")
        c.drawString(100, 640, f"Age: {record['age']}")
        c.drawString(100, 620, f"Probability: {record['probability']}%")
        c.drawString(100, 600, f"Risk Level: {record['risk_level']}")

        c.save()

        return send_file(file_name, as_attachment=True)

    except Exception as e:
        return str(e)
    
# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
