
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

USERS_FILE = DATA_DIR / "ott_users.csv"
CLUSTERED_FILE = DATA_DIR / "ott_users_clustered.csv"

GENRE_RECOMMENDATIONS = {
    "Action": ["Sci-Fi", "Drama", "Horror"],
    "Comedy": ["Romance", "Drama", "Action"],
    "Documentary": ["Drama", "Sci-Fi", "Comedy"],
    "Drama": ["Romance", "Documentary", "Sci-Fi"],
    "Horror": ["Action", "Sci-Fi", "Drama"],
    "Romance": ["Drama", "Comedy", "Documentary"],
    "Sci-Fi": ["Action", "Documentary", "Horror"],
}


def load_data():
    if not USERS_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {USERS_FILE}")

    users = pd.read_csv(USERS_FILE)

    if CLUSTERED_FILE.exists():
        clustered = pd.read_csv(CLUSTERED_FILE)

        if "customer_id" not in clustered.columns:
            raise ValueError(
                "customer_id column is missing from the clustered CSV."
            )

        if "cluster" not in clustered.columns:
            raise ValueError(
                "cluster column is missing from the clustered CSV."
            )

        users = users.drop(
            columns=["cluster"], errors="ignore"
        ).merge(
            clustered[["customer_id", "cluster"]],
            on="customer_id",
            how="left"
        )
    else:
        users["cluster"] = None

    return users


def clean_records(data):
    return data.astype(object).where(pd.notna(data), None)


@app.route("/")
def home():
    return jsonify({
        "message": "OTT Machine Learning API is running",
        "endpoints": [
            "/api/dashboard",
            "/api/segments",
            "/api/recommendations",
            "/api/customer/<customer_id>"
        ]
    })


@app.route("/api/dashboard")
def dashboard():
    users = load_data()

    total_users = len(users)

    genres = users["favorite_genre"].value_counts()

    clusters = []
    if "cluster" in users.columns:
        cluster_counts = users["cluster"].dropna().value_counts().sort_index()

        for cluster_id, count in cluster_counts.items():
            clusters.append({
                "name": f"Cluster {int(cluster_id)}",
                "cluster": int(cluster_id),
                "users": int(count)
            })

    churn_rate = None
    if "churned" in users.columns and total_users:
        churn_rate = round(
            pd.to_numeric(users["churned"], errors="coerce").mean() * 100,
            2
        )

    average_watch_hours = None
    if "watch_hours" in users.columns:
        average_watch_hours = round(
            pd.to_numeric(users["watch_hours"], errors="coerce").mean(), 2
        )

    return jsonify({
        "total_users": total_users,
        "total_genres": int(users["favorite_genre"].nunique()),
        "average_watch_hours": average_watch_hours,
        "churn_rate": churn_rate,
        "genres": [
            {"name": str(name), "users": int(count)}
            for name, count in genres.items()
        ],
        "clusters": clusters
    })


@app.route("/api/segments")
def segments():
    users = load_data()

    if users["cluster"].isna().all():
        return jsonify({
            "error": "No cluster labels found. Run clustering.py first."
        }), 400

    users["cluster"] = pd.to_numeric(
        users["cluster"], errors="coerce"
    )

    result = []

    for cluster_id, group in users.dropna(
        subset=["cluster"]
    ).groupby("cluster"):

        item = {
            "cluster": int(cluster_id),
            "name": f"Cluster {int(cluster_id)}",
            "users": int(len(group))
        }

        for column, output_name in [
            ("age", "avg_age"),
            ("watch_hours", "avg_watch_hours"),
            ("last_login_days", "avg_last_login_days"),
            ("monthly_fee", "avg_monthly_fee"),
            ("avg_watch_time_per_day", "avg_daily_watch_time"),
            ("churned", "churn_rate")
        ]:
            if column in group.columns:
                values = pd.to_numeric(
                    group[column], errors="coerce"
                )
                average = values.mean()

                item[output_name] = (
                    round(float(average), 2)
                    if pd.notna(average) else None
                )

        if "favorite_genre" in group.columns:
            common = group["favorite_genre"].mode()
            item["top_genre"] = (
                str(common.iloc[0]) if not common.empty else "Unknown"
            )

        result.append(item)

    return jsonify(result)


@app.route("/api/recommendations")
def recommendations():
    users = load_data()
    counts = users["favorite_genre"].value_counts()

    result = []

    for genre, count in counts.items():
        result.append({
            "genre": str(genre),
            "customers": int(count),
            "recommended_genres": GENRE_RECOMMENDATIONS.get(
                str(genre), []
            )
        })

    return jsonify(result)


@app.route("/api/customer/<path:customer_id>")
def customer_search(customer_id):
    users = load_data()

    matches = users[
        users["customer_id"].astype(str).str.lower()
        == customer_id.strip().lower()
    ]

    if matches.empty:
        return jsonify({"error": "Customer not found"}), 404

    customer = matches.iloc[0].to_dict()

    genre = str(customer.get("favorite_genre", ""))

    customer["recommended_genres"] = GENRE_RECOMMENDATIONS.get(
        genre, []
    )

    return jsonify(clean_records(pd.Series(customer)).to_dict())


@app.errorhandler(FileNotFoundError)
def handle_file_not_found(error):
    return jsonify({"error": str(error)}), 500


@app.errorhandler(Exception)
def handle_error(error):
    app.logger.exception("API error")
    return jsonify({"error": str(error)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
