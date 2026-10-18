import unittest

import pandas as pd
from sklearn.dummy import DummyClassifier

from churn_model import FEATURES, TARGET, _evaluate, _make_preprocessor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ChurnModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(ROOT / "data" / "ott_users.csv")

    def test_preprocessor_uses_same_prediction_feature_schema(self):
        self.assertEqual(len(FEATURES), len(set(FEATURES)))
        self.assertNotIn("customer_id", FEATURES)
        self.assertNotIn(TARGET, FEATURES)
        sample = self.data[FEATURES].iloc[:30]
        transformed = _make_preprocessor().fit_transform(sample)
        self.assertEqual(transformed.shape[0], 30)
        self.assertGreater(transformed.shape[1], len(FEATURES))

    def test_metrics_include_confusion_matrix_and_auc(self):
        x = self.data[FEATURES].iloc[:200]
        y = self.data[TARGET].iloc[:200].astype(int)
        pipeline = DummyClassifier(strategy="prior")
        pipeline.fit(x, y)
        metrics = _evaluate(pipeline, x, y)
        self.assertIn("roc_auc", metrics)
        self.assertIn("pr_auc", metrics)
        self.assertEqual(
            set(metrics["confusion_matrix"]),
            {
                "labels",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
            },
        )


if __name__ == "__main__":
    unittest.main()
