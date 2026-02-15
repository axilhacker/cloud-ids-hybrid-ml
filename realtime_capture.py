from scapy.all import sniff
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===============================
# Load Trained Models
# ===============================
print("Loading models...")

model = load_model("model_cnn.h5")
iso = joblib.load("isolation.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

print("Models loaded successfully ✅")

# ===============================
# Feature Extraction (Simplified)
# ===============================
def extract_features(packet):
    features = np.zeros(41)  # NSL-KDD has 41 features

    try:
        # Basic real-time features
        features[0] = len(packet)  # Packet length

        if packet.haslayer("TCP"):
            features[1] = 1
        if packet.haslayer("UDP"):
            features[2] = 1
        if packet.haslayer("ICMP"):
            features[3] = 1

    except:
        pass

    return features.reshape(1, -1)

# ===============================
# Detection Function
# ===============================
def detect_packet(packet):
    try:
        print("Packet Captured:", packet.summary())

        data = extract_features(packet)

        # Scale
        data_scaled = scaler.transform(data)

        # CNN reshape
        data_cnn = np.reshape(data_scaled, (1, data_scaled.shape[1], 1))

        # CNN prediction
        pred = model.predict(data_cnn, verbose=0)
        pred_class = np.argmax(pred)

        attack_type = label_encoder.inverse_transform([pred_class])[0]

        # Isolation Forest check
        iso_pred = iso.predict(data_scaled)

        if iso_pred[0] == -1:
            anomaly_status = "⚠ Anomaly Detected"
        else:
            anomaly_status = "Normal Behavior"

        print("🔍 Detected Traffic Type:", attack_type)
        print("📊 Isolation Forest:", anomaly_status)
        print("-" * 60)

    except Exception as e:
        print("Error:", e)

# ===============================
# Start Monitoring
# ===============================
print("Starting Real-Time Packet Monitoring...")
print("Press CTRL + C to stop.\n")

sniff(prn=detect_packet, store=0)
