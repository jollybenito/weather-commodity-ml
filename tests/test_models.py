"""Unit tests for models (Linear, LightGBM, XGBoost, TCN, LSTM, RegimeDetector)."""

import unittest
import numpy as np
import pandas as pd

from src.models.linear_models import RegularizedLinearModel
from src.models.tree_models import LightGBMModel, XGBoostModel
from src.models.deep_learning.tcn_model import TCNRegressor
from src.models.deep_learning.lstm_model import LSTMRegressor
from src.models.regime_detector import RegimeDetector


class TestModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 150
        self.X = pd.DataFrame({
            "feat_1": np.random.randn(n),
            "feat_2": np.random.randn(n),
            "price_vol_21d": np.abs(np.random.randn(n) * 0.15),
            "term_roll_yield": np.random.randn(n) * 0.05
        })
        self.y = pd.Series(0.5 * self.X["feat_1"] - 0.3 * self.X["feat_2"] + np.random.randn(n) * 0.1)

    def test_linear_model(self):
        m = RegularizedLinearModel(model_type="ridge", cv_folds=3)
        m.fit(self.X, self.y)
        preds = m.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_tree_models(self):
        lgb_m = LightGBMModel(n_estimators=10, max_depth=3)
        lgb_m.fit(self.X, self.y)
        preds_lgb = lgb_m.predict(self.X)
        self.assertEqual(len(preds_lgb), len(self.y))

        xgb_m = XGBoostModel(n_estimators=10, max_depth=3)
        xgb_m.fit(self.X, self.y)
        preds_xgb = xgb_m.predict(self.X)
        self.assertEqual(len(preds_xgb), len(self.y))

    def test_tcn_model(self):
        tcn = TCNRegressor(seq_len=5, epochs=2, batch_size=32)
        tcn.fit(self.X, self.y)
        preds = tcn.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_lstm_model(self):
        lstm = LSTMRegressor(seq_len=5, hidden_dim=16, epochs=2, batch_size=32)
        lstm.fit(self.X, self.y)
        preds = lstm.predict(self.X)
        self.assertEqual(len(preds), len(self.y))

    def test_regime_detector(self):
        rd = RegimeDetector(n_regimes=3, method="hmm")
        rd.fit(self.X)
        states, proba = rd.predict_regimes(self.X)
        self.assertEqual(len(states), len(self.X))
        self.assertEqual(proba.shape, (len(self.X), 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

