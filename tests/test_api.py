import csv
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import app as api


class AnalyticsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = api.app.test_client()
        cls.customer_id = str(api.customers.iloc[0]["customer_id"])

    def test_health_readiness_and_dashboard(self):
        health = self.client.get("/api/health")
        ready = self.client.get("/api/ready")
        dashboard = self.client.get("/api/dashboard")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(dashboard.status_code, 200)
        payload = dashboard.get_json()
        self.assertEqual(payload["total_customers"], len(api.customers))
        self.assertIn("data_quality", payload)
        self.assertIn("subscription_mix", payload)
        self.assertIn("churn_by_inactivity", payload)

    def test_customer_list_filters_and_pagination(self):
        response = self.client.get(
            "/api/customers?page=1&page_size=10&subscription_type=Basic"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["page_size"], 10)
        self.assertLessEqual(len(payload["customers"]), 10)
        self.assertTrue(
            all(item["subscription_type"] == "Basic" for item in payload["customers"])
        )
        self.assertIn("summary", payload)

    def test_invalid_query_inputs_use_consistent_errors(self):
        response = self.client.get("/api/customers?page=0")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "invalid_pagination")
        response = self.client.get("/api/customers?churned=maybe")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "invalid_filter")
        response = self.client.get("/api/customers?device=Hoverboard")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["code"], "invalid_filter")

    def test_customer_detail_and_not_found(self):
        found = self.client.get(f"/api/customer/{self.customer_id}")
        missing = self.client.get("/api/customer/not-a-real-customer")
        self.assertEqual(found.status_code, 200)
        self.assertEqual(found.get_json()["customer_id"], self.customer_id)
        self.assertIn("retention_suggestions", found.get_json())
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.get_json()["code"], "customer_not_found")

    def test_export_includes_filter_summary_and_filtered_records(self):
        response = self.client.get("/api/customers/export.csv?device=TV")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.content_type)
        content = response.get_data(as_text=True)
        self.assertIn("# filtered_customers,", content)
        frame = pd.read_csv(
            io.StringIO(response.get_data(as_text=True)),
            comment="#",
            skip_blank_lines=True,
        )
        self.assertFalse(frame.empty)
        self.assertTrue((frame["device"] == "TV").all())

    def test_churn_threshold_validation_and_inference(self):
        invalid = self.client.get(f"/api/churn/{self.customer_id}?threshold=1.5")
        self.assertEqual(invalid.status_code, 400)
        if api.churn_model is None:
            unavailable = self.client.get(f"/api/churn/{self.customer_id}")
            self.assertEqual(unavailable.status_code, 503)
        else:
            prediction = self.client.get(f"/api/churn/{self.customer_id}?threshold=0.4")
            self.assertEqual(prediction.status_code, 200)
            payload = prediction.get_json()
            self.assertGreaterEqual(payload["churn_risk_percent"], 0)
            self.assertLessEqual(payload["churn_risk_percent"], 100)
            self.assertIn(payload["risk_band"], {"low", "medium", "high"})
            self.assertEqual(payload["threshold"], 0.4)

    def test_recommendations_explain_genre_baseline(self):
        response = self.client.get("/api/recommendations")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json())
        self.assertIn("explanation", response.get_json()[0])
        self.assertIn("baseline", response.get_json()[0]["type"])

    def test_movie_recommendations_explain_missing_catalogue(self):
        customer = api.customers.iloc[0]
        with tempfile.TemporaryDirectory() as directory:
            missing_file = Path(directory) / "movies.csv"
            with (
                patch.object(api, "MOVIE_FILE", missing_file),
                patch.object(api, "movie_catalog", pd.DataFrame()),
                patch.object(api, "catalog_error", None),
            ):
                response = self.client.get(
                    f"/api/movie-recommendations/{customer['customer_id']}"
                )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["recommendations"], [])
        self.assertIn("data/movies_metadata.csv is missing", payload["message"])

    def test_load_catalog_parses_supplied_metadata_genres_and_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogue_file = Path(directory) / "movies_metadata.csv"
            with catalogue_file.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "title",
                        "genres",
                        "overview",
                        "original_language",
                        "release_date",
                        "vote_average",
                        "vote_count",
                    ]
                )
                writer.writerow(
                    [
                        "Sample film",
                        "[{'id': 35, 'name': 'Comedy'}, {'id': 18, 'name': 'Drama'}]",
                        "Catalogue synopsis",
                        "en",
                        "2001-05-12",
                        "7.4",
                        "120",
                    ]
                )
            with patch.object(api, "MOVIE_FILE", catalogue_file):
                catalog, error = api._load_catalog()
        self.assertIsNone(error)
        self.assertEqual(len(catalog), 1)
        self.assertEqual(catalog.iloc[0]["genre"], "Comedy, Drama")
        self.assertEqual(catalog.iloc[0]["description"], "Catalogue synopsis")
        self.assertEqual(catalog.iloc[0]["release_year"], "2001")
        self.assertEqual(catalog.iloc[0]["vote_average"], 7.4)

    def test_movie_recommendations_match_favourite_genre_and_return_metadata(self):
        customer = api.customers.iloc[0]
        favorite = str(customer["favorite_genre"])
        catalogue = pd.DataFrame(
            [
                {
                    "title": "Zeta",
                    "genre": f"Comedy, {favorite.upper()}",
                    "language": "en",
                    "release_year": 2001,
                    "description": "A catalogue synopsis",
                },
                {
                    "title": "Alpha",
                    "genre": favorite.lower(),
                    "language": "fr",
                    "release_year": 2020,
                },
                {
                    "title": "Not a match",
                    "genre": f"{favorite}land",
                    "language": "en",
                    "release_year": 2022,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            catalogue_file = Path(directory) / "movies.csv"
            catalogue_file.touch()
            with (
                patch.object(api, "MOVIE_FILE", catalogue_file),
                patch.object(api, "movie_catalog", catalogue),
                patch.object(api, "catalog_error", None),
            ):
                response = self.client.get(
                    f"/api/movie-recommendations/{customer['customer_id']}"
                )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["recommendation_type"], "exact favourite-genre catalogue match")
        self.assertEqual([movie["title"] for movie in payload["recommendations"]], ["Alpha", "Zeta"])
        self.assertEqual(payload["recommendations"][0]["release_year"], 2020)
        self.assertEqual(payload["recommendations"][1]["language"], "en")
        self.assertEqual(payload["recommendations"][1]["description"], "A catalogue synopsis")

    def test_supplied_catalogue_returns_real_exact_genre_matches(self):
        self.assertFalse(api.movie_catalog.empty)
        customer = api.customers.iloc[0]
        response = self.client.get(
            f"/api/movie-recommendations/{customer['customer_id']}"
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["catalogue_source"], "data/movies_metadata.csv")
        self.assertTrue(payload["recommendations"])
        favorite = str(customer["favorite_genre"]).casefold()
        self.assertTrue(
            all(
                favorite in api._genre_tokens(movie["genre"])
                for movie in payload["recommendations"]
            )
        )
        self.assertTrue(
            all("release_year" in movie and "language" in movie for movie in payload["recommendations"])
        )

    def test_movie_recommendations_report_invalid_catalogue(self):
        customer = api.customers.iloc[0]
        with tempfile.TemporaryDirectory() as directory:
            catalogue_file = Path(directory) / "movies.csv"
            catalogue_file.touch()
            with (
                patch.object(api, "MOVIE_FILE", catalogue_file),
                patch.object(api, "movie_catalog", pd.DataFrame()),
                patch.object(api, "catalog_error", "Movie catalogue is missing required columns: genre"),
            ):
                response = self.client.get(
                    f"/api/movie-recommendations/{customer['customer_id']}"
                )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json()["code"], "catalogue_invalid")

    def test_movie_recommendations_report_no_matching_titles_or_empty_catalogue(self):
        customer = api.customers.iloc[0]
        favorite = str(customer["favorite_genre"])
        for catalogue, expected in (
            (pd.DataFrame([{"title": "Other", "genre": f"{favorite}land"}]), "No titles"),
            (pd.DataFrame(columns=["title", "genre"]), "contains no titles"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                catalogue_file = Path(directory) / "movies.csv"
                catalogue_file.touch()
                with (
                    patch.object(api, "MOVIE_FILE", catalogue_file),
                    patch.object(api, "movie_catalog", catalogue),
                    patch.object(api, "catalog_error", None),
                ):
                    response = self.client.get(
                        f"/api/movie-recommendations/{customer['customer_id']}"
                    )
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["recommendations"], [])
            self.assertIn(expected, payload["message"])

    def test_model_metadata_exposes_persisted_version_and_evaluation(self):
        response = self.client.get("/api/model")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        if not payload["available"]:
            self.assertIsNotNone(payload["error"])
            return
        self.assertIn("version", payload["metadata"])
        self.assertIsNotNone(payload["metadata"]["trained_at"])
        self.assertIn("random_forest", payload["metadata"]["evaluation"])

    def test_invalid_reload_does_not_replace_current_dataset(self):
        current_records = len(api.customers)
        with tempfile.TemporaryDirectory() as directory:
            invalid_file = Path(directory) / "invalid.csv"
            invalid_file.write_text("customer_id,age\nc-1,40\n", encoding="utf-8")
            with patch.object(api, "CUSTOMER_FILE", invalid_file):
                response = self.client.post("/api/reload")
        self.assertEqual(response.status_code, 422)
        self.assertEqual(len(api.customers), current_records)


if __name__ == "__main__":
    unittest.main()
