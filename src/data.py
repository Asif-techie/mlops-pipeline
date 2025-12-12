"""
src/data.py

Data acquisition (API-first), fallback to local CSV, cleaning, EDA with MLflow logging.

Functions:
- download_from_uci(save_path): download dataset from UCI and save locally
- load_raw_df(): try ucimlrepo -> local -> UCI download
- clean_df(df): clean and preprocess dataframe
- perform_eda(df, save_dir): create EDA plots and log them into MLflow as a nested run
- load_heart_data(run_eda=True): top-level loader returning X, y, df
"""

import os
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

UCI_DOWNLOAD_URL = "https://archive.ics.uci.edu/dataset/45/heart+disease"
LOCAL_DATA_PATH = "data/heart.csv"
EDA_DIR = "data/eda"

COLS = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal","target"
]


def download_from_uci(save_path: str = LOCAL_DATA_PATH) -> pd.DataFrame:
    """Download the UCI processed Cleveland data and save as CSV with standard column names."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("🌐 Downloading dataset from UCI:", UCI_DOWNLOAD_URL)
    resp = requests.get(UCI_DOWNLOAD_URL, timeout=15)
    resp.raise_for_status()

    with open(save_path, "wb") as f:
        f.write(resp.content)

    df = pd.read_csv(save_path, header=None)
    df.columns = COLS
    print(f"✅ Downloaded and saved to {save_path}")
    return df


def load_raw_df() -> pd.DataFrame:
    """Try ucimlrepo -> local CSV -> download from UCI."""
    # 1) API
    try:
        print("🔌 Trying ucimlrepo.fetch_ucirepo(id=45)...")
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=45)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)

        if df.shape[1] == 14:
            df.columns = COLS

        print("✅ Loaded dataset from ucimlrepo API.")

        os.makedirs(os.path.dirname(LOCAL_DATA_PATH), exist_ok=True)
        df.to_csv(LOCAL_DATA_PATH, index=False)
        return df

    except Exception as e:
        print("⚠ ucimlrepo load failed:", e)

    # 2) Local CSV
    if os.path.exists(LOCAL_DATA_PATH):
        print("📂 Loading dataset from local CSV:", LOCAL_DATA_PATH)
        df = pd.read_csv(LOCAL_DATA_PATH, header=None)
        df.columns = COLS
        return df

    # 3) UCI download
    try:
        df = download_from_uci(LOCAL_DATA_PATH)
        return df
    except Exception as e:
        raise RuntimeError("Failed to obtain dataset from API, local file, and UCI download") from e


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe: replace ?, convert numeric, impute, binary target, drop NA."""
    df = df.copy()

    df = df.replace("?", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Mode imputation
    for col in ("ca", "thal"):
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_val)

    # Binary target
    if "target" in df.columns:
        df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

    df = df.dropna().reset_index(drop=True)
    return df


def perform_eda(df: pd.DataFrame, save_dir: str = EDA_DIR):
    """Generate histograms, heatmap, class balance plots and log them into MLflow."""
    os.makedirs(save_dir, exist_ok=True)
    print("📊 Performing EDA and logging to MLflow (nested run)...")

    with mlflow.start_run(run_name="EDA", nested=True):

        mlflow.log_param("eda_rows", df.shape[0])
        mlflow.log_param("eda_columns", df.shape[1])

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Histograms
        for col in numeric_cols:
            plt.figure(figsize=(6, 4))
            df[col].hist(bins=20)
            plt.title(f"Histogram - {col}")
            plot_path = os.path.join(save_dir, f"hist_{col}.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="eda_plots")

        # Correlation heatmap
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            corr_path = os.path.join(save_dir, "correlation_heatmap.png")
            plt.savefig(corr_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(corr_path, artifact_path="eda_plots")

        # Class Balance
        if "target" in df.columns:
            plt.figure(figsize=(5, 4))
            df["target"].value_counts().sort_index().plot(kind="bar")
            plt.title("Target Class Balance (0=Healthy,1=Disease)")
            balance_path = os.path.join(save_dir, "class_balance.png")
            plt.savefig(balance_path, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(balance_path, artifact_path="eda_plots")

    print("✅ EDA artifacts logged to MLflow (child run).")


def load_heart_data(run_eda: bool = True):
    """
    Top-level loader:
    - loads raw dataset
    - prints raw first 5 rows
    - logs raw head to MLflow
    - cleans dataset
    - prints cleaned first 2 rows
    - logs cleaned head to MLflow
    - optional EDA
    Returns X, y, df
    """

    # 1. Load raw
    raw = load_raw_df()
    print("\n📌 RAW DATA (first 5 rows):")
    print(raw.head())

    # Save raw preview to file + MLflow
    os.makedirs("data/previews", exist_ok=True)
    raw_preview_path = "data/previews/raw_head.csv"
    raw.head().to_csv(raw_preview_path, index=False)
    mlflow.log_artifact(raw_preview_path, artifact_path="data_preview")

    # 2. Clean
    df = clean_df(raw)
    print("\n🧹 CLEANED DATA (first 2 rows):")
    print(df.head(2))

    # Log cleaned preview
    clean_preview_path = "data/previews/clean_head.csv"
    df.head(2).to_csv(clean_preview_path, index=False)
    mlflow.log_artifact(clean_preview_path, artifact_path="data_preview")

    # 3. Optional EDA
    if run_eda:
        perform_eda(df)

    # 4. Split features and target
    X = df.drop(columns=["target"])
    y = df["target"].copy()

    return X, y, df
