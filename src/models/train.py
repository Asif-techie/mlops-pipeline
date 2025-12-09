"""
Train script:
- loads data
- splits
- builds pipeline
- fits and saves model artifact (joblib) to artifacts/model.pkl
"""
import os
import joblib
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.pipeline import create_pipeline

ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")

def train(save_path=MODEL_PATH, k=10):
    df = load_data()
    if "num" not in df.columns:
        raise RuntimeError("Expected 'num' target column in dataframe")

    X = df.drop(columns=["num"])
    y = df["num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = create_pipeline(k=k)
    pipeline.fit(X_train, y_train)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "X_test": X_test, "y_test": y_test}, save_path)
    print(f"Model and test split saved to: {save_path}")

if __name__ == "__main__":
    train()
