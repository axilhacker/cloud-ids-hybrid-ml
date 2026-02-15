import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess():
    df = pd.read_csv("dataset.csv")

    # Convert attack labels to binary
    df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

    X = df.drop("label", axis=1)
    y = df["label"]

    # Encode categorical
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
