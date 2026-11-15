"""Leakage-safe XGBoost workflow for the Kaggle heart-failure dataset.

Run:
    uv run python -m xgboost_ce.main

The final test split stays untouched during preprocessing, cross-validation,
and model selection. Artifacts are written to ``artifacts/heart_failure``.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer


from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

RANDOM_STATE = 42
TARGET = "HeartDisease"
EXPECTED_COLUMNS = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
    TARGET,
]


def load_and_validate(path: Path) -> pd.DataFrame:
    """Load raw data and enforce dataset assumptions."""
    df = pd.read_csv(path)
    assert list(df.columns) == EXPECTED_COLUMNS, (
        f"Unexpected schema. Expected {EXPECTED_COLUMNS}, got {list(df.columns)}"
    )
    assert df[TARGET].notna().all(), "Target contains missing values"
    assert set(df[TARGET].unique()) <= {0, 1}, "Target must contain only 0 and 1"
    assert df[TARGET].nunique() == 2, "Both target classes must be present"
    assert not df.duplicated().any(), "Duplicate rows found; inspect before training"
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target and mark known zero-as-missing measurements."""
    X = df.drop(columns=TARGET).copy()
    y = df[TARGET].astype("int8")

    for column in ["RestingBP", "Cholesterol", "MaxHR"]:
        X.loc[X[column] == 0, column] = np.nan
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = ["Sex", "ExerciseAngina", "ChestPainType", "RestingECG"]
    ordinal = ["ST_Slope"]
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[["Down", "Flat", "Up"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal,
            ),
        ],
        remainder="drop",
    )


def make_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(X)),
            ("model", estimator),
        ]
    )


def metrics_for(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }



def main(data_path: Path, artifact_dir: Path) -> None:
    df = load_and_validate(data_path)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    assert len(X_train) + len(X_test) == len(X)
    assert abs(len(X_test) / len(X) - 0.20) <= 1 / len(X)
    assert set(X_train.index).isdisjoint(X_test.index)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=3,
    )
    pipeline = make_pipeline(X_train, xgb_model)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "model__max_depth": [2, 3, 4, 5],
            "model__min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "model__learning_rate": [0.02, 0.05, 0.10],
            "model__subsample": [0.75, 0.90, 1.0],
            "model__colsample_bytree": [0.75, 0.90, 1.0],
            "model__reg_lambda": [1.0, 5.0, 20.0],
        },
        n_iter=20,
        scoring="average_precision",
        cv=cv,
        refit=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        return_train_score=True,
        verbose=10,
    )
    search.fit(X_train, y_train)

    print(f"Best parameters: {search.best_params_}")

    threshold = 0.50
    final_model = search.best_estimator_
    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": {
            "path": str(data_path),
            "rows": int(len(df)),
            "columns": EXPECTED_COLUMNS,
            "target": TARGET,
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        },
        "split": {
            "random_state": RANDOM_STATE,
            "test_fraction": 0.20,
            "training_fraction": 0.80,
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "selection": {
            "cv": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)",
            "scoring": "average_precision",
            "best_cv_score": float(search.best_score_),
            "best_params": search.best_params_,
            "decision_threshold": threshold,
            "threshold_selection": "fixed at 0.50; no validation split",
        },
        "test_metrics": {
            "xgboost": metrics_for(y_test, test_probabilities, threshold),
        },
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, artifact_dir / "heart_failure_pipeline.joblib")
    (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "features": [column for column in EXPECTED_COLUMNS if column != TARGET],
                "target": TARGET,
                "categorical_features": X.select_dtypes(exclude=["number"]).columns.tolist(),
                "numeric_features": X.select_dtypes(include=["number"]).columns.tolist(),
                "zero_to_missing": ["RestingBP", "Cholesterol", "MaxHR"],
                "decision_threshold": threshold,
            },
            indent=2,
        )
    )
    print(json.dumps(metrics["test_metrics"]["xgboost"], indent=2))
    print(f"Artifacts saved: {artifact_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("src/data/heart.csv"),
        help="Path to heart.csv",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("artifacts/heart_failure"),
        help="Output artifact directory",
    )
    args = parser.parse_args()
    main(args.data, args.artifacts)
