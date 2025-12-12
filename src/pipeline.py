"""
src/pipeline.py

Defines imbalanced pipelines and hyperparameter search spaces.
Import this from train.py.
"""

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Feature selection choices
K_CHOICES = [5, 7, 10, 12]

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
    "LogisticRegression": {"select__k": K_CHOICES, "clf__C": [0.01, 0.1, 1, 5, 10]},
    "RandomForest": {"select__k": K_CHOICES, "clf__n_estimators": [100, 200],
                     "clf__max_depth": [4, 6, None]},
    "XGBoost": {"select__k": K_CHOICES, "clf__n_estimators": [100, 200],
                "clf__max_depth": [3, 4, 5], "clf__learning_rate": [0.01, 0.05, 0.1]},
    "CatBoost": {"select__k": K_CHOICES, "clf__iterations": [200, 400],
                 "clf__depth": [3, 4, 6], "clf__learning_rate": [0.01, 0.03]}
}

search_type = {
    "LogisticRegression": GridSearchCV,
    "RandomForest": RandomizedSearchCV,
    "XGBoost": RandomizedSearchCV,
    "CatBoost": RandomizedSearchCV
}
