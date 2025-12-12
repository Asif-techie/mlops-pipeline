"""
view_mlflow_runs.py

View MLflow runs for a given experiment in terminal
with essential columns:
Run ID, Name, Accuracy, F1 Score, ROC AUC, Start Time, End Time.

Sorted by Start Time (newest first), timestamps shown in IST.
"""

import mlflow
from tabulate import tabulate
from datetime import timezone, timedelta

# -----------------------------
# Config
# -----------------------------
EXPERIMENT_NAME = "HeartDisease_Models"

# -----------------------------
# Timezone offset for IST
# -----------------------------
IST = timezone(timedelta(hours=5, minutes=30))

# -----------------------------
# Fetch experiment
# -----------------------------
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    print(f"❌ Experiment '{EXPERIMENT_NAME}' not found!")
    exit(1)

# -----------------------------
# Fetch all runs
# -----------------------------
runs = mlflow.search_runs([exp.experiment_id])
if runs.empty:
    print(f"⚠ No runs found for experiment '{EXPERIMENT_NAME}'")
    exit(0)

# -----------------------------
# Sort runs by start_time descending
# -----------------------------
runs = runs.sort_values(by="start_time", ascending=False)

# -----------------------------
# Prepare table
# -----------------------------
table = []
headers = ["Run ID", "Name", "Accuracy", "F1 Score", "ROC AUC", "Start Time (IST)", "End Time (IST)"]

for idx, row in runs.iterrows():
    start_ts = row.get("start_time", None)
    end_ts = row.get("end_time", None)

    # Convert timestamps to IST
    start_time = start_ts.tz_convert(IST).strftime("%Y-%m-%d %H:%M:%S") if start_ts is not None else "-"
    end_time = end_ts.tz_convert(IST).strftime("%Y-%m-%d %H:%M:%S") if end_ts is not None else "-"

    table.append([
        row["run_id"],
        row.get("tags.mlflow.runName", ""),
        round(row.get("metrics.accuracy", float("nan")), 3),
        round(row.get("metrics.f1_score", float("nan")), 3),
        round(row.get("metrics.roc_auc", float("nan")), 3),
        start_time,
        end_time
    ])

# -----------------------------
# Print table
# -----------------------------
print(f"\n✅ Runs for Experiment '{EXPERIMENT_NAME}' (Newest First, IST):\n")
print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
