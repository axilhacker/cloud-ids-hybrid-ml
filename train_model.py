import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("dataset_train.csv")

# Map attacks into categories
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

df["attack_type"] = df["label"].apply(map_attack)
df.drop("label", axis=1, inplace=True)

# Convert categorical columns
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes

X = df.drop("attack_type", axis=1)
y = df["attack_type"]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# ==============================
# Train RandomForest Model
# ==============================
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ==============================
# Train Isolation Forest
# ==============================
iso = IsolationForest(contamination=0.1)
iso.fit(X_scaled)

# ==============================
# Save Models
# ==============================
joblib.dump(model, "model.pkl")
joblib.dump(iso, "isolation.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

with open("accuracy.txt", "w") as f:
    f.write(str(round(accuracy, 4)))

print("Model Training Completed ✅")
print("Accuracy:", accuracy)
