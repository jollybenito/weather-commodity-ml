"""Unit tests for technological data drift analysis and splitting."""

import unittest
import numpy as np
import pandas as pd

from src.drift_analysis.data_splitter import TechnologicalEraSplitter, PurgedWalkForwardCV
from src.drift_analysis.covariate_drift import calculate_psi, CovariateDriftAnalyzer
from src.drift_analysis.concept_drift import run_chow_test


class TestDataDrift(unittest.TestCase):

    def test_psi_identical_distributions(self):
        np.random.seed(42)
        ref = pd.Series(np.random.normal(0, 1, 1000))
        cur = pd.Series(np.random.normal(0, 1, 1000))
        psi = calculate_psi(ref, cur)
        self.assertLess(psi, 0.10, "Identical distributions should have PSI < 0.10")

    def test_psi_shifted_distributions(self):
        np.random.seed(42)
        ref = pd.Series(np.random.normal(0, 1, 1000))
        cur = pd.Series(np.random.normal(2.5, 1, 1000))  # Significant shift
        psi = calculate_psi(ref, cur)
        self.assertGreaterEqual(psi, 0.25, "Shifted distribution should have PSI >= 0.25")

    def test_chow_test_structural_break(self):
        np.random.seed(42)
        # Pre-break: y = 2 * x + noise
        X_pre = np.random.normal(0, 1, (200, 1))
        y_pre = 2.0 * X_pre[:, 0] + np.random.normal(0, 0.2, 200)

        # Post-break: y = -1 * x + noise (sharp structural break)
        X_post = np.random.normal(0, 1, (200, 1))
        y_post = -1.0 * X_post[:, 0] + np.random.normal(0, 0.2, 200)

        res = run_chow_test(X_pre, y_pre, X_post, y_post)
        self.assertTrue(res["structural_break"])
        self.assertLess(res["p_value"], 0.01)

    def test_purged_walk_forward_cv_no_leakage(self):
        dates = pd.date_range("2010-01-01", "2020-12-31", freq="B")
        df = pd.DataFrame({"x": np.random.randn(len(dates))}, index=dates)
        
        cv = PurgedWalkForwardCV(n_splits=4, embargo_days=21)
        for train_idx, test_idx in cv.split(df):
            max_train_date = df.index[train_idx[-1]]
            min_test_date = df.index[test_idx[0]]
            
            diff_days = (min_test_date - max_train_date).days
            self.assertGreaterEqual(
                diff_days, 21,
                f"Embargo violated: gap is {diff_days} days, required at least 21 days."
            )


if __name__ == "__main__":
    unittest.main()

