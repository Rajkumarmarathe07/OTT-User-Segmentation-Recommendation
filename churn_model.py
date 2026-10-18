"""Reproducible churn-model comparison and evaluation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from data_quality import CHURN_FEATURES, read_customer_data


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "ott_users.csv"
MODEL_FILE = BASE_DIR / "churn_model.pkl"
METADATA_FILE = BASE_DIR / "churn_model_metadata.json"
REPORT_FILE = BASE_DIR / "reports" / "churn_evaluation.json"
TARGET = "churned"
RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURES = CHURN_FEATURES
CATEGORICAL_FEATURES = [
    "gender",
    "subscription_type",
    "region",
    "device",
    "payment_method",
    "favorite_genre",
]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _evaluate(
    estimator: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> dict[str, Any]:
    probability = estimator.predict_proba(x_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    matrix = confusion_matrix(y_test, prediction, labels=[0, 1])
    return {
        "threshold": threshold,
        "accuracy": round(float(accuracy_score(y_test, prediction)), 5),
        "precision": round(
            float(precision_score(y_test, prediction, zero_division=0)), 5
        ),
        "recall": round(float(recall_score(y_test, prediction, zero_division=0)), 5),
        "f1": round(float(f1_score(y_test, prediction, zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 5),
        "pr_auc": round(float(average_precision_score(y_test, probability)), 5),
        "confusion_matrix": {
            "labels": [0, 1],
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
    }


def _build_models(random_state: int) -> dict[str, Any]:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": Pipeline(
            [
                ("preprocessor", _make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("preprocessor", _make_preprocessor()),
                (
                    "classifier",
                    GradientBoostingClassifier(random_state=random_state),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", _make_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_evaluate(
    data_file: Path = DATA_FILE,
    model_file: Path = MODEL_FILE,
    metadata_file: Path = METADATA_FILE,
    report_file: Path = REPORT_FILE,
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
) -> dict[str, Any]:
    """Compare baseline and classifiers using one stratified holdout split."""
    data, quality = read_customer_data(data_file)
    if data is None:
        raise ValueError("Training data failed validation: " + "; ".join(quality["errors"]))
    if quality["duplicate_count"]:
        raise ValueError(
            "Customer IDs must be unique for leakage-safe evaluation; "
            f"found {quality['duplicate_count']} duplicate row(s)."
        )
    if data[TARGET].isna().any() or data[TARGET].nunique() != 2:
        raise ValueError("The churn target must contain two non-missing classes.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    x = data[FEATURES].copy()
    y = data[TARGET].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    model_reports: dict[str, Any] = {}
    models = _build_models(random_state)
    trained_models: dict[str, Any] = {}
    for name, estimator in models.items():
        estimator.fit(x_train, y_train)
        trained_models[name] = estimator
        probabilities = estimator.predict_proba(x_test)[:, 1]
        model_reports[name] = {
            "metrics": _evaluate(estimator, x_test, y_test),
            "threshold_tradeoff": {
                str(threshold): _evaluate(estimator, x_test, y_test, threshold)
                for threshold in (0.25, 0.5, 0.75)
            },
            "class_distribution": {
                "train": {
                    str(label): int(count)
                    for label, count in y_train.value_counts().sort_index().items()
                },
                "test": {
                    str(label): int(count)
                    for label, count in y_test.value_counts().sort_index().items()
                },
            },
            "probability_range": [
                round(float(probabilities.min()), 5),
                round(float(probabilities.max()), 5),
            ],
        }

    selected_model = trained_models["random_forest"]
    classifier = selected_model.named_steps["classifier"]
    feature_names = selected_model.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    feature_importance = [
        {"feature": str(name), "importance": round(float(value), 7)}
        for name, value in zip(feature_names, classifier.feature_importances_)
    ]
    feature_importance.sort(key=lambda item: item["importance"], reverse=True)
    trained_at = datetime.now(timezone.utc).isoformat()
    model_version = f"random-forest-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    report = {
        "model_version": model_version,
        "trained_at": trained_at,
        "dataset": {
            "path": str(data_file.relative_to(BASE_DIR))
            if data_file.is_relative_to(BASE_DIR)
            else data_file.name,
            "rows": int(len(data)),
            "class_distribution": {
                str(label): int(count)
                for label, count in y.value_counts().sort_index().items()
            },
        },
        "split": {
            "strategy": "stratified random train/test split",
            "test_size": test_size,
            "random_state": random_state,
            "train_records": int(len(x_train)),
            "test_records": int(len(x_test)),
        },
        "features": FEATURES,
        "leakage_review": {
            "target_excluded_from_features": TARGET not in FEATURES,
            "identifier_excluded_from_features": "customer_id" not in FEATURES,
            "timestamp_features_available": False,
            "time_based_evaluation": {
                "available": False,
                "reason": "The dataset has no usable observation or event timestamp.",
            },
            "prediction_time_note": (
                "Inputs are the non-target columns available in this customer "
                "snapshot. Their freshness at a real prediction time cannot be "
                "verified because the dataset has no observation timestamps."
            ),
        },
        "threshold_explanation": (
            "A lower threshold flags more customers: this usually increases "
            "recall and false positives. A higher threshold flags fewer "
            "customers: this usually reduces false positives but misses more "
            "churners (false negatives). Use the displayed holdout metrics to "
            "choose an operational tradeoff."
        ),
        "risk_bands": {"low_below": 0.33, "medium_below": 0.66, "high_at_or_above": 0.66},
        "models": model_reports,
        "persisted_model": "random_forest",
        "feature_importance": feature_importance[:20],
    }
    model_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, model_file)
    metadata_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    report = train_and_evaluate()
    print(f"Evaluation report saved to {REPORT_FILE}")
    print(f"Random Forest model saved to {MODEL_FILE}")
    print(f"Model version: {report['model_version']}")
    print()
    for name, model_report in report["models"].items():
        metrics = model_report["metrics"]
        print(
            f"{name:22} "
            f"accuracy={metrics['accuracy']:.3f} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} "
            f"roc_auc={metrics['roc_auc']:.3f} "
            f"pr_auc={metrics['pr_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
