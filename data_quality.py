"""Validation helpers for the customer dataset served by the application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype


REQUIRED_COLUMNS = [
    "customer_id",
    "age",
    "gender",
    "subscription_type",
    "watch_hours",
    "last_login_days",
    "region",
    "device",
    "monthly_fee",
    "churned",
    "payment_method",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "favorite_genre",
]

CHURN_FEATURES = [
    "age",
    "gender",
    "subscription_type",
    "watch_hours",
    "last_login_days",
    "region",
    "device",
    "monthly_fee",
    "payment_method",
    "number_of_profiles",
    "avg_watch_time_per_day",
    "favorite_genre",
]
CLUSTER_NUMERIC_FEATURES = [
    "age",
    "watch_hours",
    "last_login_days",
    "monthly_fee",
    "number_of_profiles",
    "avg_watch_time_per_day",
]
CLUSTER_CATEGORICAL_FEATURES = [
    "gender",
    "subscription_type",
    "region",
    "device",
    "payment_method",
    "favorite_genre",
]

NUMERIC_RANGES = {
    "age": (0, 120),
    "watch_hours": (0, None),
    "last_login_days": (0, None),
    "monthly_fee": (0, None),
    "churned": (0, 1),
    "number_of_profiles": (1, None),
    "avg_watch_time_per_day": (0, None),
}

CATEGORY_VALUES = {
    "gender": {"Female", "Male", "Other"},
    "subscription_type": {"Basic", "Premium", "Standard"},
    "region": {
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "Oceania",
        "South America",
    },
    "device": {"Desktop", "Laptop", "Mobile", "TV", "Tablet"},
    "payment_method": {
        "Credit Card",
        "Crypto",
        "Debit Card",
        "Gift Card",
        "PayPal",
    },
    "favorite_genre": {
        "Action",
        "Comedy",
        "Documentary",
        "Drama",
        "Horror",
        "Romance",
        "Sci-Fi",
    },
}


def validate_customer_data(
    data: pd.DataFrame | None,
    source_error: str | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable quality report for customer records."""
    if data is None:
        return {
            "status": "invalid",
            "record_count": 0,
            "missing_value_count": 0,
            "duplicate_count": 0,
            "columns": [],
            "errors": [source_error or "Customer data is unavailable."],
            "warnings": [],
        }

    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        errors.append(
            "Required customer columns are missing: "
            + ", ".join(missing_columns)
        )

    missing_value_count = int(data.isna().sum().sum())
    duplicate_count = (
        int(data["customer_id"].duplicated(keep="first").sum())
        if "customer_id" in data
        else 0
    )
    if missing_value_count:
        warnings.append(f"{missing_value_count} missing cell(s) found.")
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate customer ID row(s) found.")

    columns: list[dict[str, Any]] = []
    for column in data.columns:
        missing = int(data[column].isna().sum())
        column_report: dict[str, Any] = {
            "name": str(column),
            "dtype": str(data[column].dtype),
            "missing": missing,
            "invalid": 0,
            "unexpected_categories": [],
        }

        if column in NUMERIC_RANGES:
            if not is_numeric_dtype(data[column]):
                column_report["invalid"] = int(
                    pd.to_numeric(data[column], errors="coerce").isna().sum()
                    - missing
                )
                if column_report["invalid"]:
                    errors.append(
                        f"Column '{column}' contains non-numeric values."
                    )
                else:
                    warnings.append(
                        f"Column '{column}' is stored as {data[column].dtype}; "
                        "numeric values are parseable."
                    )
            values = pd.to_numeric(data[column], errors="coerce")
            lower, upper = NUMERIC_RANGES[column]
            invalid_range = values.notna()
            if lower is not None:
                invalid_range &= values < lower
            if upper is not None:
                invalid_range |= values > upper
            column_report["invalid"] += int(invalid_range.sum())
            if invalid_range.any():
                warnings.append(
                    f"Column '{column}' has {int(invalid_range.sum())} "
                    "value(s) outside its accepted range."
                )

        if column in CATEGORY_VALUES:
            unexpected = sorted(
                set(data[column].dropna().astype(str)) - CATEGORY_VALUES[column]
            )
            column_report["unexpected_categories"] = unexpected
            if unexpected:
                warnings.append(
                    f"Column '{column}' contains unexpected categories: "
                    + ", ".join(unexpected[:10])
                )

        if column == "customer_id":
            blank_ids = int(data[column].astype("string").str.strip().eq("").sum())
            missing_ids = int(data[column].isna().sum())
            if blank_ids or missing_ids:
                column_report["invalid"] += blank_ids
                errors.append(
                    f"{blank_ids + missing_ids} customer ID(s) are missing or blank."
                )

        columns.append(column_report)

    status = "invalid" if errors else ("warning" if warnings else "valid")
    return {
        "status": status,
        "record_count": int(len(data)),
        "missing_value_count": missing_value_count,
        "duplicate_count": duplicate_count,
        "columns": columns,
        "errors": errors,
        "warnings": warnings,
    }


def read_customer_data(path: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Read customer CSV and validate its structure, reporting file errors."""
    if not path.is_file():
        error = f"Required customer file not found: {path}"
        return None, validate_customer_data(None, error)
    try:
        data = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeError) as error:
        message = f"Could not read customer file '{path}': {error}"
        return None, validate_customer_data(None, message)
    report = validate_customer_data(data)
    if report["errors"]:
        return None, report
    return data, report
