from __future__ import annotations

import csv
import ast
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_quality import (
    CHURN_FEATURES,
    CLUSTER_CATEGORICAL_FEATURES,
    CLUSTER_NUMERIC_FEATURES,
    read_customer_data,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CUSTOMER_FILE = DATA_DIR / "ott_users.csv"
CLUSTER_FILE = DATA_DIR / "ott_users_clustered.csv"
SUMMARY_FILE = DATA_DIR / "cluster_summary.csv"
PREPROCESSED_FILE = DATA_DIR / "ott_users_preprocessed.csv"
MOVIE_FILE = DATA_DIR / "movies_metadata.csv"
MODEL_FILE = BASE_DIR / "churn_model.pkl"
MODEL_METADATA_FILE = BASE_DIR / "churn_model_metadata.json"

FEATURES = CHURN_FEATURES
GENRE_MAPPING = {
    "Action": ["Sci-Fi", "Drama", "Horror"],
    "Comedy": ["Romance", "Drama", "Action"],
    "Documentary": ["Drama", "Sci-Fi", "Comedy"],
    "Drama": ["Romance", "Documentary", "Sci-Fi"],
    "Horror": ["Action", "Sci-Fi", "Drama"],
    "Romance": ["Drama", "Comedy", "Documentary"],
    "Sci-Fi": ["Action", "Documentary", "Horror"],
}
NUMERIC_CLUSTER_FEATURES = CLUSTER_NUMERIC_FEATURES
CATEGORICAL_CLUSTER_FEATURES = CLUSTER_CATEGORICAL_FEATURES
DEFAULT_RISK_BANDS = {"low_below": 0.33, "medium_below": 0.66}

app = Flask(__name__)
logging.getLogger("werkzeug").disabled = True
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
CORS(app, resources={r"/api/*": {"origins": allowed_origins}})

customers = pd.DataFrame()
clustered = pd.DataFrame()
cluster_summary = pd.DataFrame()
movie_catalog = pd.DataFrame()
churn_model: Any = None
model_metadata: dict[str, Any] = {
    "version": "unavailable",
    "trained_at": None,
    "features": FEATURES,
    "evaluation": None,
}
data_quality: dict[str, Any] = {
    "status": "invalid",
    "record_count": 0,
    "missing_value_count": 0,
    "duplicate_count": 0,
    "columns": [],
    "errors": ["Customer data has not been loaded."],
    "warnings": [],
}
data_error: str | None = None
model_error: str | None = None
catalog_error: str | None = None
_risk_scores: np.ndarray | None = None
_cluster_assignments: dict[str, int] = {}
_segment_names: dict[int, str] | None = None
_retention_cutoffs: tuple[float, float] | None = None


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_model() -> tuple[Any, dict[str, Any], str | None]:
    if not MODEL_FILE.is_file():
        return None, {
            "version": "unavailable",
            "trained_at": None,
            "features": FEATURES,
            "evaluation": None,
        }, f"Churn model file not found: {MODEL_FILE.name}"
    try:
        loaded_model = joblib.load(MODEL_FILE)
    except (OSError, EOFError, ValueError, ModuleNotFoundError) as error:
        return None, {
            "version": "unavailable",
            "trained_at": None,
            "features": FEATURES,
            "evaluation": None,
        }, f"Could not load churn model: {error}"

    metadata: dict[str, Any] = {
        "version": "legacy-unversioned",
        "trained_at": None,
        "features": list(getattr(loaded_model, "feature_names_in_", FEATURES)),
        "evaluation": None,
        "importance": [],
        "importance_note": (
            "Global model feature importance is descriptive, not a "
            "customer-specific causal explanation."
        ),
    }
    metadata_error = None
    if MODEL_METADATA_FILE.is_file():
        try:
            with MODEL_METADATA_FILE.open(encoding="utf-8") as file:
                saved_metadata = json.load(file)
            if isinstance(saved_metadata, dict):
                metadata.update(saved_metadata)
                metadata["version"] = saved_metadata.get(
                    "model_version",
                    saved_metadata.get("version", metadata["version"]),
                )
                metadata["evaluation"] = saved_metadata.get(
                    "models", saved_metadata.get("evaluation")
                )
                metadata["importance"] = saved_metadata.get(
                    "feature_importance", metadata.get("importance", [])
                )
        except (OSError, json.JSONDecodeError) as error:
            metadata_error = f"Could not read model metadata: {error}"
    classifier = (
        loaded_model.named_steps.get("classifier")
        if hasattr(loaded_model, "named_steps")
        else loaded_model
    )
    if not metadata.get("importance") and hasattr(classifier, "feature_importances_"):
        transformer = (
            loaded_model.named_steps.get("preprocessor")
            if hasattr(loaded_model, "named_steps")
            else None
        )
        names = (
            transformer.get_feature_names_out().tolist()
            if transformer is not None and hasattr(transformer, "get_feature_names_out")
            else []
        )
        importance = [
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(names, classifier.feature_importances_)
        ]
        importance.sort(key=lambda item: item["importance"], reverse=True)
        metadata["importance"] = importance[:20]
    return loaded_model, metadata, metadata_error


def _load_catalog() -> tuple[pd.DataFrame, str | None]:
    if not MOVIE_FILE.is_file():
        return pd.DataFrame(), None
    try:
        catalog = pd.read_csv(MOVIE_FILE, low_memory=False).fillna("")
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        return pd.DataFrame(), f"Could not read movie catalogue: {error}"
    if "title" not in catalog.columns:
        return pd.DataFrame(), "Movie catalogue is missing required column: title"

    if "genre" not in catalog.columns and "genres" in catalog.columns:
        def genre_names(raw_genres: Any) -> str:
            if not raw_genres:
                return ""
            try:
                entries = ast.literal_eval(str(raw_genres))
            except (SyntaxError, ValueError):
                return ""
            if not isinstance(entries, list):
                return ""
            return ", ".join(
                str(entry["name"]).strip()
                for entry in entries
                if isinstance(entry, dict)
                and isinstance(entry.get("name"), str)
                and entry["name"].strip()
            )

        catalog["genre"] = catalog["genres"].map(genre_names)
    if "genre" not in catalog.columns:
        return pd.DataFrame(), (
            "Movie catalogue is missing required column: genre "
            "(or the source 'genres' field)."
        )

    source_fields = {
        "title": "title",
        "genre": "genre",
        "overview": "description",
        "original_language": "language",
        "release_date": "release_date",
        "original_title": "original_title",
        "vote_average": "vote_average",
        "vote_count": "vote_count",
        "popularity": "popularity",
        "runtime": "runtime",
    }
    canonical = catalog[
        [field for field in source_fields if field in catalog.columns]
    ].rename(columns=source_fields)
    if "release_date" in canonical.columns:
        canonical["release_year"] = pd.to_datetime(
            canonical["release_date"], errors="coerce"
        ).dt.year.astype("Int64").astype("string").fillna("")
        canonical = canonical.drop(columns=["release_date"])
    canonical["title"] = canonical["title"].astype(str).str.strip()
    canonical["genre"] = canonical["genre"].astype(str).str.strip()
    canonical = canonical[
        canonical["title"].ne("") & canonical["genre"].ne("")
    ].reset_index(drop=True)
    if canonical.empty:
        return canonical, "Movie catalogue has no usable rows with both title and genre."
    return canonical, None


def _load_bundle() -> tuple[
    pd.DataFrame | None,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    Any,
    dict[str, Any],
    str | None,
    pd.DataFrame,
    str | None,
    str | None,
]:
    loaded_customers, quality = read_customer_data(CUSTOMER_FILE)
    loaded_clustered = pd.DataFrame()
    loaded_summary = pd.DataFrame()
    if loaded_customers is not None:
        loaded_clustered = _read_optional_csv(CLUSTER_FILE)
        loaded_summary = _read_optional_csv(SUMMARY_FILE)
        if "customer_id" not in loaded_clustered or "cluster" not in loaded_clustered:
            loaded_clustered = pd.DataFrame()
        if not loaded_summary.empty and "cluster" not in loaded_summary:
            loaded_summary = pd.DataFrame()
    loaded_model, metadata, loaded_model_error = _load_model()
    catalog, loaded_catalog_error = _load_catalog()
    return (
        loaded_customers,
        quality,
        loaded_clustered,
        loaded_summary,
        loaded_model,
        metadata,
        loaded_model_error,
        catalog,
        loaded_catalog_error,
        quality["errors"][0] if quality["errors"] else None,
    )


def _apply_bundle(bundle: tuple[Any, ...]) -> None:
    global customers, data_quality, clustered, cluster_summary
    global churn_model, model_metadata, model_error
    global movie_catalog, catalog_error
    global data_error, _risk_scores, _cluster_assignments, _segment_names
    global _retention_cutoffs
    (
        loaded_customers,
        quality,
        loaded_clustered,
        loaded_summary,
        loaded_model,
        metadata,
        loaded_model_error,
        catalog,
        loaded_catalog_error,
        loaded_data_error,
    ) = bundle
    customers = loaded_customers if loaded_customers is not None else pd.DataFrame()
    data_quality = quality
    clustered = loaded_clustered
    cluster_summary = loaded_summary
    churn_model = loaded_model
    model_metadata = metadata
    model_error = loaded_model_error
    movie_catalog = catalog
    catalog_error = loaded_catalog_error
    data_error = loaded_data_error
    _risk_scores = None
    _cluster_assignments = {}
    if (
        not clustered.empty
        and "customer_id" in clustered
        and "cluster" in clustered
    ):
        for customer_id, cluster_id in zip(
            clustered["customer_id"].astype(str), clustered["cluster"]
        ):
            if pd.notna(cluster_id):
                _cluster_assignments[customer_id] = int(cluster_id)
    _segment_names = None
    _retention_cutoffs = None


def _data_ready() -> bool:
    return not customers.empty and data_quality["status"] != "invalid"


def _require_data() -> tuple[Any, int] | None:
    if not _data_ready():
        message = data_error or "Customer data is not ready."
        return _error(message, 503, "data_not_ready", data_quality)
    return None


def _error(
    message: str,
    status: int,
    code: str,
    details: Any | None = None,
) -> tuple[Any, int]:
    body: dict[str, Any] = {"error": message, "code": code}
    if details is not None:
        body["details"] = details
    return jsonify(body), status


def _customer_cluster_map() -> dict[str, int]:
    return _cluster_assignments


def _segment_profiles() -> list[dict[str, Any]]:
    if clustered.empty or "cluster" not in clustered:
        return []
    assignments = clustered[["customer_id", "cluster"]].copy()
    assignments["customer_id"] = assignments["customer_id"].astype(str)
    source = customers.copy()
    source["customer_id"] = source["customer_id"].astype(str)
    data = source.merge(assignments, on="customer_id", how="inner")
    if data.empty:
        return []
    profiles = []
    median_watch = float(customers["watch_hours"].median())
    median_recency = float(customers["last_login_days"].median())
    for cluster_id, group in data.groupby("cluster", sort=True):
        avg_watch = float(group["watch_hours"].mean())
        avg_recency = float(group["last_login_days"].mean())
        activity_name = "Higher viewing" if avg_watch >= median_watch else "Lower viewing"
        recency_name = "less recent" if avg_recency >= median_recency else "more recent"
        plan_mix = (
            group["subscription_type"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .rename_axis("plan")
            .reset_index(name="percent")
            .to_dict(orient="records")
        )
        genre_mix = (
            group["favorite_genre"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
            .rename_axis("genre")
            .reset_index(name="percent")
            .to_dict(orient="records")
        )
        profiles.append(
            {
                "cluster": int(cluster_id),
                "name": f"{activity_name} · {recency_name}",
                "users": int(len(group)),
                "avg_age": round(float(group["age"].mean()), 2),
                "avg_watch_hours": round(avg_watch, 2),
                "avg_last_login_days": round(avg_recency, 2),
                "avg_monthly_fee": round(float(group["monthly_fee"].mean()), 2),
                "avg_daily_watch_time": round(
                    float(group["avg_watch_time_per_day"].mean()), 2
                ),
                "churn_rate": round(float(group["churned"].mean() * 100), 2),
                "top_genre": str(group["favorite_genre"].mode().iloc[0]),
                "plan_mix": plan_mix,
                "genre_mix": genre_mix,
            }
        )
    return profiles


def _get_risk_scores() -> np.ndarray | None:
    global _risk_scores
    if churn_model is None:
        return None
    if _risk_scores is not None:
        return _risk_scores
    missing_features = [feature for feature in FEATURES if feature not in customers]
    if missing_features:
        raise ValueError(
            "Customer data is missing model input features: "
            + ", ".join(missing_features)
        )
    if not hasattr(churn_model, "predict_proba"):
        raise ValueError("The loaded churn model does not provide risk probabilities.")
    probabilities = churn_model.predict_proba(customers[FEATURES])
    classes = list(
        churn_model.named_steps["classifier"].classes_
        if hasattr(churn_model, "named_steps")
        else churn_model.classes_
    )
    if 1 not in classes:
        raise ValueError("The loaded churn model has no positive churn class.")
    _risk_scores = np.asarray(probabilities[:, classes.index(1)], dtype=float)
    return _risk_scores


def _risk_band(probability: float) -> str:
    if probability < DEFAULT_RISK_BANDS["low_below"]:
        return "low"
    if probability < DEFAULT_RISK_BANDS["medium_below"]:
        return "medium"
    return "high"


def _genre_tokens(value: Any) -> set[str]:
    return {
        token.strip().casefold()
        for token in re.split(r"[,|;/&]+", str(value))
        if token.strip()
    }


def _segment_name_map() -> dict[int, str]:
    global _segment_names
    if _segment_names is None:
        _segment_names = {
            item["cluster"]: item["name"] for item in _segment_profiles()
        }
    return _segment_names


def _retention_suggestions(customer: pd.Series) -> list[str]:
    global _retention_cutoffs
    if _retention_cutoffs is None:
        _retention_cutoffs = (
            float(customers["watch_hours"].quantile(0.25)),
            float(customers["last_login_days"].quantile(0.75)),
        )
    lower_watch, upper_inactivity = _retention_cutoffs
    suggestions = []
    if float(customer["watch_hours"]) <= lower_watch:
        suggestions.append(
            "Viewing is at or below the dataset's 25th percentile "
            f"({lower_watch:.1f} hours); review content discovery options."
        )
    if float(customer["last_login_days"]) >= upper_inactivity:
        suggestions.append(
            "Inactivity is at or above the dataset's 75th percentile "
            f"({upper_inactivity:.0f} days); consider an opt-in re-engagement "
            "message."
        )
    if not suggestions:
        suggestions.append(
            "No low-viewing or high-inactivity signal was triggered by the "
            "current percentile rules."
        )
    return suggestions


def _customer_payload(
    row: pd.Series,
    threshold: float = 0.5,
    risk_scores: np.ndarray | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    payload = row.to_dict()
    customer_id = str(payload["customer_id"])
    cluster_map = _customer_cluster_map()
    if customer_id in cluster_map and pd.notna(cluster_map[customer_id]):
        payload["cluster"] = int(cluster_map[customer_id])
        payload["segment_name"] = _segment_name_map().get(payload["cluster"])
    else:
        payload["cluster"] = None
        payload["segment_name"] = None

    probability = (
        float(risk_scores[index])
        if risk_scores is not None and index is not None
        else None
    )
    payload.update(
        {
            "churn_risk_percent": (
                round(probability * 100, 2) if probability is not None else None
            ),
            "risk_band": _risk_band(probability) if probability is not None else None,
            "churn_prediction": (
                int(probability >= threshold) if probability is not None else None
            ),
            "classification_threshold": threshold,
            "risk_band_cutoffs": DEFAULT_RISK_BANDS,
            "retention_suggestions": _retention_suggestions(row),
            "recommended_genres": GENRE_MAPPING.get(
                str(payload.get("favorite_genre", "")), []
            ),
            "recommendation_type": "transparent_favorite_genre_baseline",
        }
    )
    return payload


def _filtered_customer_indices() -> tuple[pd.DataFrame, list[int], float] | tuple[None, Any, None]:
    threshold_raw = request.args.get("threshold", "0.5")
    try:
        threshold = float(threshold_raw)
    except ValueError:
        return None, _error("threshold must be a number from 0 to 1.", 400, "invalid_threshold"), None
    if not 0 <= threshold <= 1:
        return None, _error("threshold must be a number from 0 to 1.", 400, "invalid_threshold"), None

    data = customers.copy()
    data["customer_id"] = data["customer_id"].astype(str)
    data["_source_index"] = np.arange(len(data))
    query = request.args.get("q", "").strip().casefold()
    if query:
        searchable = data.astype(str).apply(
            lambda column: column.str.casefold().str.contains(query, regex=False)
        )
        data = data[searchable.any(axis=1)]

    categorical_filters = {
        "subscription_type": "subscription_type",
        "region": "region",
        "device": "device",
        "favorite_genre": "favorite_genre",
    }
    for argument, column in categorical_filters.items():
        value = request.args.get(argument)
        if value:
            allowed = set(customers[column].dropna().astype(str).str.casefold())
            if value.casefold() not in allowed:
                return None, _error(
                    f"Unsupported {argument} filter value.",
                    400,
                    "invalid_filter",
                ), None
            data = data[data[column].astype(str).str.casefold() == value.casefold()]

    inactivity_band = request.args.get("inactivity_band")
    inactivity_ranges = {
        "0-7": (0, 7),
        "8-14": (8, 14),
        "15-30": (15, 30),
        ">30": (31, None),
    }
    if inactivity_band:
        if inactivity_band not in inactivity_ranges:
            return None, _error(
                "Unsupported inactivity_band filter.", 400, "invalid_filter"
            ), None
        lower, upper = inactivity_ranges[inactivity_band]
        data = data[data["last_login_days"] >= lower]
        if upper is not None:
            data = data[data["last_login_days"] <= upper]

    watch_band = request.args.get("watch_band")
    watch_ranges = {
        "0-5": (0, 5),
        ">5-10": (5, 10),
        ">10-20": (10, 20),
        ">20-40": (20, 40),
        ">40": (40, None),
    }
    if watch_band:
        if watch_band not in watch_ranges:
            return None, _error("Unsupported watch_band filter.", 400, "invalid_filter"), None
        lower, upper = watch_ranges[watch_band]
        data = data[data["watch_hours"] >= lower] if lower == 0 else data[data["watch_hours"] > lower]
        if upper is not None:
            data = data[data["watch_hours"] <= upper]

    churn_filter = request.args.get("churned")
    if churn_filter not in (None, ""):
        if churn_filter not in {"0", "1"}:
            return None, _error("churned must be 0 or 1.", 400, "invalid_filter"), None
        data = data[data["churned"].astype(int) == int(churn_filter)]

    cluster_map = _customer_cluster_map()
    segment_filter = request.args.get("segment")
    if segment_filter not in (None, ""):
        try:
            segment_id = int(segment_filter)
        except ValueError:
            return None, _error("segment must be an integer cluster ID.", 400, "invalid_filter"), None
        data = data[
            data["customer_id"].map(cluster_map).fillna(-1).astype(int) == segment_id
        ]

    scores = None
    risk_filter = request.args.get("risk_band")
    sort_by = request.args.get("sort_by", "customer_id")
    if risk_filter or sort_by == "risk":
        try:
            scores = _get_risk_scores()
        except ValueError as error:
            return None, _error(str(error), 500, "prediction_failed"), None
        if scores is None:
            return None, _error(
                model_error or "Churn model is not available.",
                503,
                "model_not_ready",
            ), None
    if risk_filter:
        if risk_filter not in {"low", "medium", "high"}:
            return None, _error(
                "risk_band must be low, medium, or high.", 400, "invalid_filter"
            ), None
        matching_indices = data["_source_index"].to_numpy(dtype=int)
        data = data[
            np.array([_risk_band(float(scores[i])) == risk_filter for i in matching_indices])
        ]

    data = data.drop(columns=["_source_index"])
    return data, [], threshold


def _load_initial_state() -> None:
    try:
        _apply_bundle(_load_bundle())
    except (OSError, pd.errors.ParserError, ValueError) as error:
        global data_error
        data_error = f"Could not load project data: {error}"
        data_quality.update(
            {
                "status": "invalid",
                "errors": [data_error],
            }
        )


_load_initial_state()


@app.get("/")
def home():
    return jsonify(
        {
            "message": "OTT customer analytics API is running",
            "endpoints": [
                "/api/health",
                "/api/ready",
                "/api/dashboard",
                "/api/customers",
                "/api/customers/export.csv",
                "/api/segments",
                "/api/segments/evaluation",
                "/api/recommendations",
                "/api/customer/<customer_id>",
                "/api/churn/<customer_id>",
                "/api/model",
                "/api/reload",
            ],
        }
    )


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/ready")
def readiness():
    body = {
        "status": "ready" if _data_ready() else "not_ready",
        "data": data_quality,
        "model_available": churn_model is not None,
        "model_error": model_error,
        "catalogue_available": not movie_catalog.empty,
        "catalogue_error": catalog_error,
    }
    return jsonify(body), (200 if _data_ready() else 503)


@app.post("/api/reload")
def reload_data():
    try:
        bundle = _load_bundle()
    except (OSError, pd.errors.ParserError, ValueError) as error:
        return _error(f"Could not reload project data: {error}", 500, "reload_failed")
    if bundle[0] is None or bundle[1]["status"] == "invalid":
        return _error(
            bundle[1]["errors"][0] if bundle[1]["errors"] else "Customer data is invalid.",
            422,
            "data_validation_failed",
            bundle[1],
        )
    _apply_bundle(bundle)
    return jsonify(
        {
            "status": "reloaded",
            "data": data_quality,
            "model_available": churn_model is not None,
            "model_error": model_error,
            "catalogue_available": not movie_catalog.empty,
            "catalogue_error": catalog_error,
        }
    )


@app.get("/api/dashboard")
def dashboard():
    unavailable = _require_data()
    if unavailable:
        return unavailable
    total_customers = int(len(customers))
    churn_rate = float(customers["churned"].mean() * 100)
    active_customers = int((customers["last_login_days"] <= 7).sum())
    genre_counts = customers["favorite_genre"].value_counts()
    subscription_counts = customers["subscription_type"].value_counts()
    segment_profiles = _segment_profiles()

    watch_bins = pd.cut(
        customers["watch_hours"],
        bins=[-0.001, 5, 10, 20, 40, float("inf")],
        labels=["0–5", ">5–10", ">10–20", ">20–40", ">40"],
    )
    watch_distribution = (
        watch_bins.value_counts(sort=False)
        .rename_axis("range")
        .reset_index(name="customers")
        .to_dict(orient="records")
    )
    inactivity_bins = pd.cut(
        customers["last_login_days"],
        bins=[-0.001, 7, 14, 30, float("inf")],
        labels=["0–7 days", "8–14 days", "15–30 days", ">30 days"],
    )
    inactivity = customers.assign(inactivity_band=inactivity_bins)
    churn_by_inactivity = (
        inactivity.groupby("inactivity_band", observed=False)
        .agg(customers=("churned", "size"), churn_rate=("churned", "mean"))
        .reset_index()
    )
    churn_by_inactivity["churn_rate"] = (
        churn_by_inactivity["churn_rate"].mul(100).round(2)
    )
    churn_by_segment = [
        {
            "name": item["name"],
            "cluster": item["cluster"],
            "customers": item["users"],
            "churn_rate": item["churn_rate"],
        }
        for item in segment_profiles
    ]

    return jsonify(
        {
            "total_customers": total_customers,
            "total_users": total_customers,
            "total_genres": int(customers["favorite_genre"].nunique()),
            "average_watch_hours": round(float(customers["watch_hours"].mean()), 2),
            "active_customers": active_customers,
            "active_definition": "last_login_days <= 7",
            "churn_rate": round(churn_rate, 2),
            "segment_count": len(segment_profiles),
            "genres": [
                {"name": str(name), "users": int(count)}
                for name, count in genre_counts.items()
            ],
            "subscription_mix": [
                {"name": str(name), "customers": int(count)}
                for name, count in subscription_counts.items()
            ],
            "watch_distribution": watch_distribution,
            "churn_by_segment": churn_by_segment,
            "churn_by_inactivity": churn_by_inactivity.rename(
                columns={"inactivity_band": "name"}
            ).to_dict(orient="records"),
            "segments": segment_profiles,
            "data_quality": data_quality,
            "model": {
                "available": churn_model is not None,
                "version": model_metadata.get("version"),
                "trained_at": model_metadata.get("trained_at"),
                "features": model_metadata.get("features", FEATURES),
                "evaluation": model_metadata.get("evaluation"),
                "error": model_error,
            },
            "limitations": {
                "date_filters": "Unavailable: the dataset has no event or observation dates.",
                "churn_by_inactivity": (
                    "Descriptive association only; it does not prove inactivity "
                    "causes churn."
                ),
            },
        }
    )


@app.get("/api/segments")
def segments():
    unavailable = _require_data()
    if unavailable:
        return unavailable
    return jsonify(_segment_profiles())


@app.get("/api/segments/evaluation")
def evaluate_segments():
    unavailable = _require_data()
    if unavailable:
        return unavailable
    if not PREPROCESSED_FILE.is_file():
        return _error(
            "Preprocessed customer data is required for candidate k evaluation.",
            503,
            "segmentation_data_missing",
        )
    try:
        prepared = pd.read_csv(PREPROCESSED_FILE)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        return _error(f"Could not read preprocessed data: {error}", 500, "data_read_failed")
    features = prepared.drop(columns=["customer_id", "cluster"], errors="ignore")
    if features.empty or len(features) < 4:
        return _error("Not enough customer records for k evaluation.", 422, "insufficient_data")
    results = []
    for k in range(2, min(9, len(features))):
        estimator = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = estimator.fit_predict(features)
        silhouette = float(silhouette_score(features, labels))
        results.append(
            {"k": k, "inertia": round(float(estimator.inertia_), 3), "silhouette": round(silhouette, 4)}
        )
    return jsonify(
        {
            "candidates": results,
            "selected_cluster_source": "data/ott_users_clustered.csv",
            "features": [column for column in prepared.columns if column != "customer_id"],
            "identifier_excluded": True,
            "interpretation": (
                "Candidate scores compare compactness and separation; segment "
                "differences are descriptive and do not establish causes."
            ),
        }
    )


@app.get("/api/recommendations")
def recommendations():
    unavailable = _require_data()
    if unavailable:
        return unavailable
    result = []
    for genre, count in customers["favorite_genre"].value_counts().items():
        suggested = GENRE_MAPPING.get(str(genre), [])
        result.append(
            {
                "genre": str(genre),
                "customers": int(count),
                "recommended_genres": suggested,
                "type": "transparent genre mapping baseline",
                "explanation": (
                    f"Based on a fixed, inspectable mapping from the customer's "
                    f"favourite genre ({genre}); it is not learned from viewing "
                    "interactions."
                ),
                "item_recommendations_available": not movie_catalog.empty,
            }
        )
    return jsonify(result)


@app.get("/api/customers")
def list_customers():
    unavailable = _require_data()
    if unavailable:
        return unavailable
    filtered, issue, threshold = _filtered_customer_indices()
    if filtered is None:
        return issue

    try:
        page = int(request.args.get("page", "1"))
        page_size = int(request.args.get("page_size", "25"))
    except ValueError:
        return _error("page and page_size must be integers.", 400, "invalid_pagination")
    if page < 1 or page_size not in {10, 25, 50, 100}:
        return _error(
            "page must be at least 1 and page_size must be 10, 25, 50, or 100.",
            400,
            "invalid_pagination",
        )

    sort_by = request.args.get("sort_by", "customer_id")
    sort_columns = {
        "customer_id": "customer_id",
        "age": "age",
        "watch_hours": "watch_hours",
        "last_login_days": "last_login_days",
        "churned": "churned",
        "subscription_type": "subscription_type",
        "region": "region",
        "device": "device",
        "favorite_genre": "favorite_genre",
        "risk": "churn_risk_percent",
    }
    if sort_by not in sort_columns:
        return _error("Unsupported sort_by field.", 400, "invalid_sort")
    ascending = request.args.get("order", "asc").lower()
    if ascending not in {"asc", "desc"}:
        return _error("order must be asc or desc.", 400, "invalid_sort")

    total = len(filtered)
    filtered = filtered.copy()
    if sort_by == "risk":
        scores = _get_risk_scores()
        filtered["churn_risk_percent"] = [
            float(scores[index]) * 100
            for index in _filtered_source_indices(filtered)
        ]
        filtered = filtered.sort_values(
            "churn_risk_percent", ascending=ascending == "asc"
        )
    else:
        filtered = filtered.sort_values(
            sort_columns[sort_by], ascending=ascending == "asc", kind="stable"
        )
    start = (page - 1) * page_size
    page_rows = filtered.iloc[start : start + page_size]
    risk_scores = _get_risk_scores()
    source_indices = _filtered_source_indices(page_rows)
    records = [
        _customer_payload(row, threshold, risk_scores, source_index)
        for (_, row), source_index in zip(page_rows.iterrows(), source_indices)
    ]
    churned_count = int(filtered["churned"].sum())
    summary = {
        "customers": total,
        "churned": churned_count,
        "churn_rate": round(churned_count / total * 100, 2) if total else 0,
        "average_watch_hours": (
            round(float(filtered["watch_hours"].mean()), 2) if total else 0
        ),
    }
    return jsonify(
        {
            "customers": records,
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": (total + page_size - 1) // page_size,
            "summary": summary,
            "threshold": threshold,
            "risk_band_cutoffs": DEFAULT_RISK_BANDS,
            "model_available": churn_model is not None,
        }
    )


def _filtered_source_indices(page_rows: pd.DataFrame) -> list[int]:
    # Customer IDs are unique in the supplied data; use the source frame to
    # preserve row identity for risk probabilities after sorting/filtering.
    id_to_index = {
        str(customer_id): index
        for index, customer_id in enumerate(customers["customer_id"].astype(str))
    }
    return [id_to_index[str(customer_id)] for customer_id in page_rows["customer_id"]]


@app.get("/api/customers/export.csv")
def export_customers():
    unavailable = _require_data()
    if unavailable:
        return unavailable
    filtered, issue, threshold = _filtered_customer_indices()
    if filtered is None:
        return issue
    risk_scores = _get_risk_scores()
    source_indices = _filtered_source_indices(filtered)
    records = [
        _customer_payload(row, threshold, risk_scores, source_index)
        for (_, row), source_index in zip(filtered.iterrows(), source_indices)
    ]
    export_frame = pd.DataFrame(records)
    total = len(export_frame)
    summary = {
        "filtered_customers": total,
        "churned_customers": int(export_frame["churned"].sum()) if total else 0,
        "churn_rate_percent": (
            round(float(export_frame["churned"].mean() * 100), 2) if total else 0
        ),
        "average_watch_hours": (
            round(float(export_frame["watch_hours"].mean()), 2) if total else 0
        ),
    }
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["# OTT customer export summary"])
    for name, value in summary.items():
        writer.writerow([f"# {name}", value])
    writer.writerow([])
    export_frame.to_csv(output, index=False)
    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv; charset=utf-8"
    response.headers["Content-Disposition"] = 'attachment; filename="filtered_customers.csv"'
    return response


@app.get("/api/customer/<customer_id>")
def get_customer(customer_id: str):
    unavailable = _require_data()
    if unavailable:
        return unavailable
    if not customer_id.strip() or len(customer_id) > 128:
        return _error("customer_id must be between 1 and 128 characters.", 400, "invalid_customer_id")
    matches = customers[customers["customer_id"].astype(str) == customer_id]
    if matches.empty:
        return _error("Customer not found.", 404, "customer_not_found")
    if len(matches) > 1:
        return _error(
            "Customer ID is duplicated; correct the source data before using this record.",
            409,
            "duplicate_customer_id",
        )
    try:
        scores = _get_risk_scores()
    except ValueError as error:
        return _error(str(error), 500, "prediction_failed")
    source_index = int(matches.index[0])
    # Model scores follow DataFrame row position rather than arbitrary index labels.
    source_position = customers.index.get_loc(source_index)
    payload = _customer_payload(
        matches.iloc[0], 0.5, scores, int(source_position) if scores is not None else None
    )
    payload["model"] = model_metadata
    payload["explanation"] = {
        "method": "global model feature importance",
        "features": model_metadata.get("importance", [])[:5],
        "note": (
            "These are model-wide predictors and are not a customer-specific "
            "causal explanation."
        ),
    }
    payload["prediction_caveat"] = (
        "Risk is a model estimate, not a guarantee of future customer behavior."
    )
    return jsonify(payload)


@app.get("/api/churn/<customer_id>")
def predict_customer_churn(customer_id: str):
    unavailable = _require_data()
    if unavailable:
        return unavailable
    if churn_model is None:
        return _error(
            model_error or "Churn model is not available.",
            503,
            "model_not_ready",
        )
    try:
        threshold = float(request.args.get("threshold", "0.5"))
    except ValueError:
        return _error("threshold must be a number from 0 to 1.", 400, "invalid_threshold")
    if not 0 <= threshold <= 1:
        return _error("threshold must be a number from 0 to 1.", 400, "invalid_threshold")
    matches = customers[customers["customer_id"].astype(str) == customer_id]
    if matches.empty:
        return _error("Customer not found.", 404, "customer_not_found")
    if len(matches) > 1:
        return _error("Customer ID is duplicated.", 409, "duplicate_customer_id")
    try:
        scores = _get_risk_scores()
    except ValueError as error:
        return _error(str(error), 500, "prediction_failed")
    index = int(customers.index.get_loc(matches.index[0]))
    probability = float(scores[index])
    classifier = (
        churn_model.named_steps.get("classifier")
        if hasattr(churn_model, "named_steps")
        else churn_model
    )
    feature_importance = model_metadata.get("importance", [])
    if not feature_importance and hasattr(classifier, "feature_importances_"):
        transformer = churn_model.named_steps.get("preprocessor")
        names = (
            transformer.get_feature_names_out().tolist()
            if transformer is not None and hasattr(transformer, "get_feature_names_out")
            else []
        )
        feature_importance = [
            {"feature": str(name), "importance": float(value)}
            for name, value in zip(names, classifier.feature_importances_)
        ]
        feature_importance.sort(key=lambda item: item["importance"], reverse=True)
        feature_importance = feature_importance[:5]
    return jsonify(
        {
            "customer_id": customer_id,
            "churn_prediction": int(probability >= threshold),
            "churn_risk_percent": round(probability * 100, 2),
            "risk_band": _risk_band(probability),
            "risk_band_cutoffs": DEFAULT_RISK_BANDS,
            "threshold": threshold,
            "status": "Likely to churn" if probability >= threshold else "Likely to stay",
            "model_version": model_metadata.get("version"),
            "explanation": {
                "method": "global model feature importance",
                "features": feature_importance[:5],
                "note": (
                    "These are model-wide predictors, not a personalized causal "
                    "explanation. No SHAP dependency is installed."
                ),
            },
            "caveat": "This estimate does not mean the customer will definitely churn.",
        }
    )


@app.get("/api/model")
def get_model_metadata():
    return jsonify(
        {
            "available": churn_model is not None,
            "error": model_error,
            "metadata": model_metadata,
        }
    )


@app.get("/api/movie-recommendations/<customer_id>")
def recommend_customer_movies(customer_id: str):
    unavailable = _require_data()
    if unavailable:
        return unavailable
    customer = customers[customers["customer_id"].astype(str) == customer_id]
    if customer.empty:
        return _error("Customer not found.", 404, "customer_not_found")
    favorite_genre = str(customer.iloc[0]["favorite_genre"])
    if catalog_error:
        return _error(catalog_error, 503, "catalogue_invalid")
    if not MOVIE_FILE.is_file():
        return jsonify(
            {
                "customer_id": customer_id,
                "favorite_genre": favorite_genre,
                "available": False,
                "recommendations": [],
                "message": (
                    "Movie catalogue data/movies_metadata.csv is missing. No "
                    "titles are being suggested."
                ),
            }
        )

    filtered = movie_catalog.copy()
    supported_filters = {
        "genre": "genre",
        "language": "language",
        "maturity_rating": "maturity_rating",
        "release_year": "release_year",
    }
    available_filters = [
        parameter
        for parameter, column in supported_filters.items()
        if column in filtered.columns
    ]
    filter_options = {}
    for parameter, column in supported_filters.items():
        if column not in filtered.columns:
            continue
        if parameter == "genre":
            filter_options[parameter] = sorted(
                {
                    token
                    for values in filtered[column]
                    for token in _genre_tokens(values)
                }
            )
        else:
            filter_options[parameter] = sorted(
                filtered[column].astype(str).unique().tolist()
            )
    for parameter in request.args:
        if parameter not in supported_filters:
            return _error(
                f"Unsupported catalogue filter '{parameter}'.",
                400,
                "invalid_filter",
            )
    for parameter, column in supported_filters.items():
        value = request.args.get(parameter)
        if value:
            if column not in filtered.columns:
                return _error(
                    f"Catalogue does not support the '{parameter}' filter.",
                    400,
                    "unsupported_catalogue_filter",
                )
            if parameter == "release_year":
                try:
                    year = int(value)
                except ValueError:
                    return _error("release_year must be an integer.", 400, "invalid_filter")
                filtered = filtered[
                    pd.to_numeric(filtered[column], errors="coerce") == year
                ]
            else:
                if parameter == "genre":
                    filtered = filtered[
                        filtered[column].map(
                            lambda genres: value.casefold() in _genre_tokens(genres)
                        )
                    ]
                else:
                    filtered = filtered[
                        filtered[column].astype(str).str.casefold() == value.casefold()
                    ]
    if movie_catalog.empty:
        return jsonify(
            {
                "customer_id": customer_id,
                "favorite_genre": favorite_genre,
                "available": True,
                "supported_filters": available_filters,
                "filter_options": filter_options,
                "catalogue_source": "data/movies_metadata.csv",
                "recommendations": [],
                "message": "The movie catalogue contains no titles.",
            }
        )
    matched = filtered[
        filtered["genre"].map(
            lambda genres: favorite_genre.casefold() in _genre_tokens(genres)
        )
    ]
    if matched.empty:
        return jsonify(
            {
                "customer_id": customer_id,
                "favorite_genre": favorite_genre,
                "available": True,
                "supported_filters": available_filters,
                "filter_options": filter_options,
                "catalogue_source": "data/movies_metadata.csv",
                "recommendations": [],
                "message": (
                    f"No titles in data/movies_metadata.csv match the favourite genre "
                    f"'{favorite_genre}' and the current filters."
                ),
            }
        )
    movie_columns = list(movie_catalog.columns)
    results = matched.sort_values("title", kind="stable").head(5)
    return jsonify(
        {
            "customer_id": customer_id,
            "favorite_genre": favorite_genre,
            "available": True,
            "supported_filters": available_filters,
            "filter_options": filter_options,
            "catalogue_source": "data/movies_metadata.csv",
            "recommendation_type": "exact favourite-genre catalogue match",
            "explanation": (
                f"Titles and metadata come from data/movies_metadata.csv. "
                f"Genre is matched against the customer's favourite genre "
                f"({favorite_genre}); this catalogue does not confirm streaming "
                f"availability."
            ),
            "recommendations": results[movie_columns].to_dict(orient="records"),
        }
    )


@app.errorhandler(404)
def not_found(error):
    del error
    return _error("Resource not found.", 404, "not_found")


@app.errorhandler(405)
def method_not_allowed(error):
    del error
    return _error("HTTP method is not allowed for this endpoint.", 405, "method_not_allowed")


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=False, port=5000)
