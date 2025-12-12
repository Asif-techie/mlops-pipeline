"""
src/train.py

Main training script:
 - wraps entire process in a parent MLflow run
 - calls data.load_heart_data(run_eda=True) (EDA will run as a nested MLflow child run)
 - logs start and end timestamps in human-readable format
 - for each pipeline, performs hyperparameter search and logs model, metrics, artifacts in nested runs
 - dynamically adjusts n_iter for RandomizedSearchCV to avoid UserWarnings
"""

import os
import tempfile
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from pipeline import pipelines, param_spaces, search_type
from data import load_heart_data
from utils_plot import save_cm, save_roc
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from math import prod
import datetime

# Experiment configuration
EXPERIMENT_NAME = "HeartDisease_Models"
mlflow.set_experiment(EXPERIMENT_NAME)

N_JOBS = -1
CV_FOLDS = 5
RANDOM_STATE = 42


def total_param_combinations(param_grid):
    """Calculate total number of parameter combinations."""
    sizes = [len(v) for v in param_grid.values()]
    return prod(sizes) if sizes else 1


def main():
    # Parent run
    parent_start = datetime.datetime.now()
    with mlflow.start_run(run_name="Main_Training_Run") as parent_run:
        mlflow.log_param("main_start_time", parent_start.strftime("%Y-%m-%d %H:%M:%S"))

        # Load data and run EDA (nested MLflow run)
        X, y, _ = load_heart_data(run_eda=True)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        mlflow.log_param("train_rows", X_train.shape[0])
        mlflow.log_param("test_rows", X_test.shape[0])

        # Iterate over pipelines
        for name, pipe in pipelines.items():
            print(f"\n🔹 Starting hyperparameter search for {name}")
            Searcher = search_type.get(name)
            params = param_spaces.get(name, {})

            if Searcher is None:
                print(f"⚠ No searcher defined for {name}. Skipping.")
                continue

            # Determine n_iter dynamically to avoid warning
            if Searcher == GridSearchCV:
                search = Searcher(
                    pipe,
                    params,
                    cv=CV_FOLDS,
                    n_jobs=N_JOBS,
                    scoring="roc_auc"
                )
            else:
                max_combos = total_param_combinations(params)
                n_iter = min(25, max_combos)  # dynamically adjust n_iter
                search = Searcher(
                    pipe,
                    params,
                    n_iter=n_iter,
                    cv=CV_FOLDS,
                    n_jobs=N_JOBS,
                    scoring="roc_auc",
                    random_state=RANDOM_STATE
                )

            # Nested MLflow run for each model
            model_start = datetime.datetime.now()
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("start_time", model_start.strftime("%Y-%m-%d %H:%M:%S"))

                search.fit(X_train, y_train)
                best_model = search.best_estimator_

                # Predictions & scores
                y_pred = best_model.predict(X_test)
                try:
                    y_score = best_model.predict_proba(X_test)[:, 1]
                except Exception:
                    try:
                        y_score = best_model.decision_function(X_test)
                    except Exception:
                        y_score = y_pred

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                roc = roc_auc_score(y_test, y_score)

                # Log best params
                best_params = search.best_params_
                for k, v in best_params.items():
                    mlflow.log_param(k, v)

                # Log metrics
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc)

                # Save artifacts
                tmpdir = tempfile.mkdtemp()
                cm_path = os.path.join(tmpdir, f"{name}_cm.png")
                save_cm(y_test, y_pred, cm_path)
                mlflow.log_artifact(cm_path, artifact_path="artifacts")

                roc_path = os.path.join(tmpdir, f"{name}_roc.png")
                save_roc(y_test, y_score, roc_path)
                mlflow.log_artifact(roc_path, artifact_path="artifacts")

                # Log the model
                mlflow.sklearn.log_model(best_model, artifact_path=f"models/{name}")

                # Log end timestamp
                model_end = datetime.datetime.now()
                mlflow.log_param("end_time", model_end.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"{name} → Accuracy={acc:.3f}, F1={f1:.3f}, ROC_AUC={roc:.3f}")
                print(f"Run started at: {model_start}, ended at: {model_end}")

        # Parent run end time
        parent_end = datetime.datetime.now()
        mlflow.log_param("main_end_time", parent_end.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"\n✅ All runs completed. Main run started at {parent_start}, ended at {parent_end}")


if __name__ == "__main__":
    main()
