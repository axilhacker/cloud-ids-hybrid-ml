from flask import Flask, render_template
import pandas as pd
import numpy as np
import joblib
import os
from pymongo import MongoClient

app = Flask(__name__)

# ==============================
# LOAD MODEL & SCALER
# ==============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==============================
# LOAD TEST DATASET
# ==============================
test_df = pd.read_csv("dataset_test.csv")

# ==============================
# MONGODB CONNECTION
# ==============================
MONGO_URI = os.environ.get("MONGO_URI")
collection = None

if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI)
        db = client["cloud_ids"]
        collection = db["predictions"]
        print("MongoDB Connected ✅")
    except Exception as e:
        print("MongoDB connection error:", e)
else:
    print("Running without MongoDB")

# ==============================
# HOME ROUTE
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

# ==============================
# PREDICT ROUTE
# ==============================
@app.route("/predict")
def predict():

    # Pick random row
    random_row = test_df.sample(1)

    X = random_row.drop("label", axis=1)
    y_actual = random_row["label"].values[0]

    # 🔥 Encode categorical columns safely
    categorical_columns = ["protocol_type", "service", "flag"]

    for col in categorical_columns:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes

    # Convert to numeric safely
    X = X.apply(pd.to_numeric)

    # Scale
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)[0]

    predicted_attack = label_encoder.inverse_transform([prediction])[0]
    actual_attack = label_encoder.inverse_transform([y_actual])[0]

    status = "Correct ✅" if predicted_attack == actual_attack else "Incorrect ❌"

    # Save to MongoDB safely
    if collection is not None:
        try:
            collection.insert_one({
                "predicted": predicted_attack,
                "actual": actual_attack,
                "status": status
            })
        except Exception as e:
            print("MongoDB insert error:", e)

    return render_template(
        "index.html",
        prediction=f"Predicted: {predicted_attack}",
        actual=f"Actual: {actual_attack}",
        status=status
    )

# ==============================
# RUN APP
# ==============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
