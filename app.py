from flask import Flask, render_template
import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
from pymongo import MongoClient

app = Flask(__name__)

# ==============================
# Load Trained Models
# ==============================
model = load_model("model_cnn.h5")
iso = joblib.load("isolation.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==============================
# Load Accuracy
# ==============================
with open("accuracy.txt") as f:
    accuracy = f.read()

# ==============================
# MongoDB Cloud Connection
# ==============================
MONGO_URI = os.environ.get("MONGO_URI")  # Set this in Render environment

if MONGO_URI:
    client = MongoClient(MONGO_URI)
    db = client["cloud_ids"]
    collection = db["attack_logs"]
else:
    collection = None  # If Mongo not configured

# ==============================
# Load Test Dataset
# ==============================
test_df = pd.read_csv("dataset_test.csv")

def map_attack(label):
    if label == "normal":
        return "Normal"
    elif label in ["back","land","neptune","pod","smurf","teardrop"]:
        return "DoS"
    elif label in ["ipsweep","nmap","portsweep","satan"]:
        return "Probe"
    elif label in ["ftp_write","guess_passwd","imap","multihop","phf","spy","warezclient","warezmaster"]:
        return "R2L"
    else:
        return "U2R"

test_df["attack_type"] = test_df["label"].apply(map_attack)
test_df.drop("label", axis=1, inplace=True)

X_test = test_df.drop("attack_type", axis=1)

for col in X_test.select_dtypes(include=['object']).columns:
    X_test[col] = X_test[col].astype('category').cat.codes


# ==============================
# Routes
# ==============================
@app.route('/')
def home():
    return render_template("index.html", accuracy=accuracy)


@app.route('/predict')
def predict():

    random_sample = test_df.sample(1)

    actual_attack = random_sample["attack_type"].values[0]
    X_sample = random_sample.drop("attack_type", axis=1)

    data_scaled = scaler.transform(X_sample)
    data_cnn = np.reshape(data_scaled, (1, data_scaled.shape[1], 1))

    pred = model.predict(data_cnn, verbose=0)
    pred_class = np.argmax(pred)

    predicted_attack = label_encoder.inverse_transform([pred_class])[0]

    if actual_attack == predicted_attack:
        status = "✅ Correct Detection"
    else:
        status = "❌ Incorrect Detection"

    # Save to MongoDB Cloud
    if collection:
        collection.insert_one({
            "actual_attack": actual_attack,
            "predicted_attack": predicted_attack,
            "status": status
        })

    return render_template("index.html",
                           prediction=f"Predicted: {predicted_attack}",
                           actual=f"Actual: {actual_attack}",
                           status=status,
                           accuracy=accuracy)


# ==============================
# Run for Cloud Deployment
# ==============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
