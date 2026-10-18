import unittest
from pathlib import Path

import pandas as pd

from data_quality import read_customer_data, validate_customer_data


ROOT = Path(__file__).resolve().parents[1]


class CustomerDataQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data" / "ott_users.csv")

    def test_project_dataset_is_valid(self):
        report = validate_customer_data(self.data)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["record_count"], 5000)
        self.assertEqual(report["missing_value_count"], 0)
        self.assertEqual(report["duplicate_count"], 0)

    def test_required_columns_are_reported(self):
        report = validate_customer_data(self.data.drop(columns=["favorite_genre"]))
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("favorite_genre" in message for message in report["errors"])
        )

    def test_missing_customer_id_blocks_readiness(self):
        data = self.data.iloc[:3].copy()
        data.loc[data.index[0], "customer_id"] = None
        report = validate_customer_data(data)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(any("customer ID" in message for message in report["errors"]))

    def test_duplicate_customer_ids_are_counted(self):
        data = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        report = validate_customer_data(data)
        self.assertEqual(report["duplicate_count"], 1)
        self.assertEqual(report["status"], "warning")

    def test_invalid_numeric_value_is_reported(self):
        data = self.data.iloc[:3].copy()
        data["age"] = data["age"].astype(str)
        data.loc[data.index[0], "age"] = "unknown"
        report = validate_customer_data(data)
        self.assertEqual(report["status"], "invalid")
        self.assertTrue(
            any("age" in message and "non-numeric" in message for message in report["errors"])
        )

    def test_unexpected_category_and_range_are_reported(self):
        data = self.data.iloc[:3].copy()
        data.loc[data.index[0], "device"] = "Unknown device"
        data.loc[data.index[1], "age"] = 140
        report = validate_customer_data(data)
        self.assertEqual(report["status"], "warning")
        device = next(column for column in report["columns"] if column["name"] == "device")
        self.assertEqual(device["unexpected_categories"], ["Unknown device"])
        self.assertTrue(
            any("outside its accepted range" in message for message in report["warnings"])
        )

    def test_missing_source_file_has_clear_error(self):
        data, report = read_customer_data(ROOT / "data" / "missing.csv")
        self.assertIsNone(data)
        self.assertEqual(report["status"], "invalid")
        self.assertIn("Required customer file not found", report["errors"][0])


if __name__ == "__main__":
    unittest.main()
