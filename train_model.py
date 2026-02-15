import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("dataset_train.csv")

# ==============================
# Map Attack Categories
# ==============================
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

# ==============================
# Encode Labels
# ==============================
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["attack_type"])
joblib.dump(label_encoder, "label_encoder.pkl")

X = df.drop("attack_type", axis=1)

# Encode categorical features
for col in X.select_dtypes(include=['object']).columns:
    X[col] = LabelEncoder().fit_transform(X[col])

# ==============================
# Scale Features
# ==============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# ==============================
# CNN Model
# ==============================
X_cnn = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_scaled.shape[1],1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_cnn, y, epochs=3, batch_size=64)

model.save("model_cnn.h5")

# ==============================
# Evaluation
# ==============================
predictions = model.predict(X_cnn)
predicted_classes = np.argmax(predictions, axis=1)

accuracy = accuracy_score(y, predicted_classes)
print("Model Accuracy:", accuracy)

with open("accuracy.txt", "w") as f:
    f.write(str(accuracy))

cm = confusion_matrix(y, predicted_classes)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")

# ==============================
# Isolation Forest
# ==============================
iso = IsolationForest(contamination=0.1)
iso.fit(X_scaled)
joblib.dump(iso, "isolation.pkl")

print("Model Training Completed ✅")
