
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from data_quality import (
    CLUSTER_CATEGORICAL_FEATURES,
    CLUSTER_NUMERIC_FEATURES,
    read_customer_data,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "ott_users.csv"
OUTPUT_FILE = BASE_DIR / "data" / "ott_users_preprocessed.csv"

df, quality = read_customer_data(DATA_FILE)
if df is None:
    raise ValueError("Customer data failed validation: " + "; ".join(quality["errors"]))
if quality["duplicate_count"]:
    raise ValueError(
        "Customer IDs must be unique before preprocessing; "
        f"found {quality['duplicate_count']} duplicate row(s)."
    )

print("Original dataset shape:", df.shape)

numerical_features = CLUSTER_NUMERIC_FEATURES
categorical_features = CLUSTER_CATEGORICAL_FEATURES

X = df[numerical_features + categorical_features].copy()

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_processed = preprocessor.fit_transform(X)

feature_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray")
    else X_processed,
    columns=feature_names
)

processed_df.insert(0, "customer_id", df["customer_id"].values)

processed_df.to_csv(OUTPUT_FILE, index=False)

print("\nPreprocessing completed successfully!")
print("Processed dataset shape:", processed_df.shape)
print("\nMissing values:", processed_df.isnull().sum().sum())
print("\nProcessed data preview:")
print(processed_df.head())

print(f"\nSaved to {OUTPUT_FILE}")
