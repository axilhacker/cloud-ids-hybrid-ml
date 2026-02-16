from flask import Flask, render_template, redirect, url_for
import numpy as np
import pandas as pd
import joblib
import random
import os
from pymongo import MongoClient

app = Flask(__name__)

# =========================
# LOAD MODEL FILES
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =========================
# MONGODB CONNECTION
# =========================
MONGO_URI = os.environ.get("MONGO_URI")

collection = None

if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client["cloud_ids"]
        collection = db["predictions"]
        print("MongoDB connected successfully ✅")
    except Exception as e:
        print("MongoDB connection failed:", e)
else:
    print("MONGO_URI not set. Running without database.")

# =========================
# LOAD TEST DATA
# =========================
test_df = pd.read_csv("dataset_test.csv")

# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICT ROUTE
# =========================
@app.route("/predict")
def predict():

    # Select random row
    random_row = test_df.sample(1)

    X = random_row.drop("label", axis=1)
    y_actual = random_row["label"].values[0]

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    predicted_attack = label_encoder.inverse_transform([prediction])[0]
    actual_attack = label_encoder.inverse_transform([y_actual])[0]

    status = "Correct ✅" if predicted_attack == actual_attack else "Incorrect ❌"

    # =========================
    # SAVE TO MONGODB (SAFE)
    # =========================
    if collection is not None:
        try:
            collection.insert_one({
                "predicted": predicted_attack,
                "actual": actual_attack,
                "status": status
            })
        except Exception as e:
            print("Mongo insert error:", e)

    return render_template(
        "index.html",
        prediction=f"Predicted: {predicted_attack}",
        actual=f"Actual: {actual_attack}",
        status=status
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
