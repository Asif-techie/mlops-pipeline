"""
Data loading for UCI Heart Disease dataset.
Provides `load_data()` that returns a cleaned DataFrame.
"""
import pandas as pd

def load_heart_data():
    df = pd.read_csv("data/heart.csv", header=None)

    df.columns = [
        "age","sex","cp","trestbps","chol","fbs",
        "restecg","thalach","exang","oldpeak",
        "slope","ca","thal","target"
    ]

    # Replace ? with NaN
    df = df.replace("?", pd.NA)

    # Convert everything numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop missing rows
    df = df.dropna()

    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y



