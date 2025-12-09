# src/train.py
# src/train.py
# src/train.py
# ==========================
# File: src/train.py
# ==========================

import time
import os
import tempfile
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# ==========================
# Helper functions for plotting
# ==========================
def save_cm(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha='center', va='center', color='red')
    ax.set_title("Confusion Matrix")
    fig.savefig(path)
    plt.close(fig)

def save_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_title("ROC Curve")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)

# ==========================
# Load and preprocess data
# ==========================
def load_data():
    heart = fetch_ucirepo(id=45)
    X = heart.data.features
    y = heart.data.targets
    df = pd.concat([X, y], axis=1)

    df.replace("?", np.nan, inplace=True)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    for col in ["ca", "thal"]:
        if col in df.columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

    if "num" in df.columns:
        df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    X = df.drop(columns=["num"])
    y = df["num"]

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train, X_test, y_train, y_test = load_data()
print("✅ Data ready:", X_train.shape, X_test.shape)

from data import load_heart_data

def load_data():
    X, y = load_heart_data()
    return train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================
# Define pipelines and hyperparameter spaces
# ==========================
K_choices = [5, 7, 10, 12]

pipelines = {
    "LogisticRegression": ImbPipeline([
        ("select", SelectKBest(f_classif)),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "RandomForest": ImbPipeline([
        ("select", SelectKBest(f_classif)),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(random_state=42))
    ]),
    "XGBoost": ImbPipeline([
        ("select", SelectKBest(f_classif)),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42))
    ]),
    "CatBoost": ImbPipeline([
        ("select", SelectKBest(f_classif)),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("clf", CatBoostClassifier(verbose=0, random_state=42))
    ])
}

param_spaces = {
    "LogisticRegression": {"select__k": K_choices, "clf__C": [0.01,0.1,1,5,10]},
    "RandomForest": {"select__k": K_choices, "clf__n_estimators":[100,200], "clf__max_depth":[4,6,None]},
    "XGBoost": {"select__k": K_choices, "clf__n_estimators":[100,200], "clf__max_depth":[3,4,5], "clf__learning_rate":[0.01,0.05,0.1]},
    "CatBoost": {"select__k": K_choices, "clf__iterations":[200,400], "clf__depth":[3,4,6], "clf__learning_rate":[0.01,0.03]}
}

search_type = {
    "LogisticRegression": GridSearchCV,
    "RandomForest": RandomizedSearchCV,
    "XGBoost": RandomizedSearchCV,
    "CatBoost": RandomizedSearchCV
}

# ==========================
# MLflow experiment
# ==========================
EXPERIMENT_NAME = "HeartDisease_Models"
mlflow.set_experiment(EXPERIMENT_NAME)

N_JOBS = -1
CV_FOLDS = 5

# ==========================
# Run hyperparameter search and log to MLflow
# ==========================
for name, pipe in pipelines.items():
    print(f"\n🔹 Hyperparameter search for {name}")
    Searcher = search_type[name]
    if Searcher == GridSearchCV:
        search = Searcher(pipe, param_spaces[name], cv=CV_FOLDS, n_jobs=N_JOBS, scoring="roc_auc")
    else:
        search = Searcher(pipe, param_spaces[name], n_iter=25, cv=CV_FOLDS, n_jobs=N_JOBS, scoring="roc_auc", random_state=42)

    with mlflow.start_run(run_name=name):
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # Metrics
        y_pred = best_model.predict(X_test)
        try:
            y_score = best_model.predict_proba(X_test)[:,1]
        except:
            y_score = y_pred

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_score)

        # Log params and metrics
        for k, v in search.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # Log artifacts
        tmpdir = tempfile.mkdtemp()
        cm_path = os.path.join(tmpdir, f"{name}_cm.png")
        save_cm(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        roc_path = os.path.join(tmpdir, f"{name}_roc.png")
        save_roc(y_test, y_score, roc_path)
        mlflow.log_artifact(roc_path)

        # Log model
        mlflow.sklearn.log_model(best_model, artifact_path=f"models/{name}")

        print(f"{name} → Accuracy={acc:.3f}, F1={f1:.3f}, ROC_AUC={roc:.3f}")

# ==========================
# Show all MLflow runs
# ==========================
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
runs = mlflow.search_runs([exp.experiment_id])
print("\n=== MLflow Experiment Summary ===")
for idx, row in runs.iterrows():
    print(f"Run ID: {row['run_id']}, Accuracy: {row['metrics.accuracy']:.3f}, "
          f"F1: {row['metrics.f1_score']:.3f}, ROC_AUC: {row['metrics.roc_auc']:.3f}")
