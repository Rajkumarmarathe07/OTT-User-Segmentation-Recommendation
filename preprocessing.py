
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Load dataset
df = pd.read_csv("data/ott_users.csv")

print("Original dataset shape:", df.shape)

# 2. Remove duplicate rows
df = df.drop_duplicates()

# 3. Remove rows without a customer ID
df = df.dropna(subset=["customer_id"])

# 4. Select useful features
numerical_features = [
    "age",
    "watch_hours",
    "last_login_days",
    "monthly_fee",
    "number_of_profiles",
    "avg_watch_time_per_day"
]

categorical_features = [
    "gender",
    "subscription_type",
    "region",
    "device",
    "payment_method",
    "favorite_genre"
]

# 5. Separate input features
X = df[numerical_features + categorical_features].copy()

# 6. Define preprocessing pipelines
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

# 7. Transform the data
X_processed = preprocessor.fit_transform(X)

# 8. Convert transformed data to a DataFrame
feature_names = preprocessor.get_feature_names_out()

processed_df = pd.DataFrame(
    X_processed.toarray() if hasattr(X_processed, "toarray")
    else X_processed,
    columns=feature_names
)

# Preserve customer IDs for identification
processed_df.insert(0, "customer_id", df["customer_id"].values)

# 9. Save processed dataset
processed_df.to_csv(
    "data/ott_users_preprocessed.csv",
    index=False
)

print("\nPreprocessing completed successfully!")
print("Processed dataset shape:", processed_df.shape)
print("\nMissing values:", processed_df.isnull().sum().sum())
print("\nProcessed data preview:")
print(processed_df.head())

print("\nSaved to data/ott_users_preprocessed.csv")
