"""The Killer Experiment: 5-Stage Ablation Suite & Model Bake-off.

Executes:
1. Algorithm Bake-off:
   Linear (Ridge) vs LightGBM vs XGBoost vs TCN vs LSTM
2. 5-Way Ablation Benchmark:
   - Exp 1: Price-Only
   - Exp 2: Weather-Only
   - Exp 3: Price + Weather
   - Exp 4: Price + Weather + Term Structure
   - Exp 5: Price + Weather + Term Structure + Regime (Meta-Model)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import logging

from src.feature_engineering.panel_builder import PanelDataset
from src.drift_analysis.data_splitter import PurgedWalkForwardCV
from src.models.linear_models import RegularizedLinearModel
from src.models.tree_models import LightGBMModel, XGBoostModel
from src.models.deep_learning.tcn_model import TCNRegressor
from src.models.deep_learning.lstm_model import LSTMRegressor
from src.models.regime_detector import RegimeDetector
from src.meta_model.adaptive_allocator import AdaptiveSignalAllocator, MetaLearner
from src.meta_model.backtester import InstitutionalBacktester, BacktestMetrics

logger = logging.getLogger(__name__)


def evaluate_model_family(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> np.ndarray:
    """Trains a model family on train set and generates predictions on test set."""
    if model_name == "Linear_Ridge":
        model = RegularizedLinearModel(model_type="ridge")
        model.fit(X_train, y_train)
        return model.predict(X_test)
    elif model_name == "LightGBM":
        model = LightGBMModel(n_estimators=100, learning_rate=0.04, max_depth=4)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    elif model_name == "XGBoost":
        model = XGBoostModel(n_estimators=100, learning_rate=0.04, max_depth=4)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    elif model_name == "TCN":
        model = TCNRegressor(seq_len=10, epochs=15, batch_size=64)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    elif model_name == "LSTM":
        model = LSTMRegressor(seq_len=10, hidden_dim=24, epochs=15, batch_size=64)
        model.fit(X_train, y_train)
        return model.predict(X_test)
    else:
        raise ValueError(f"Unknown model family {model_name}")


class ExperimentSuiteRunner:
    """Orchestrates model bake-off and the 5 ablation experiments."""

    def __init__(
        self,
        panel_dataset: PanelDataset,
        tx_cost_bps: float = 5.0,
        n_cv_splits: int = 5
    ):
        self.dataset = panel_dataset
        self.backtester = InstitutionalBacktester(transaction_cost_bps=tx_cost_bps)
        self.cv = PurgedWalkForwardCV(n_splits=n_cv_splits, embargo_days=21)

    def run_algorithm_bakeoff(self) -> pd.DataFrame:
        """Compares Linear, LightGBM, XGBoost, TCN, and LSTM on the full feature set."""
        X, y = self.dataset.get_X_y("all")
        algorithms = ["Linear_Ridge", "LightGBM", "XGBoost", "TCN", "LSTM"]
        
        results = []
        for algo in algorithms:
            logger.info(f"Running walk-forward evaluation for {algo}...")
            all_preds = []
            all_actuals = []

            for train_idx, test_idx in self.cv.split(X):
                X_tr, y_tr = X.iloc[train_idx].values, y.iloc[train_idx].values
                X_te, y_te = X.iloc[test_idx].values, y.iloc[test_idx].values

                preds = evaluate_model_family(X_tr, y_tr, X_te, y_te, algo)
                all_preds.extend(preds)
                all_actuals.extend(y_te)

            pred_s = pd.Series(all_preds)
            act_s = pd.Series(all_actuals)
            metrics, _ = self.backtester.run_backtest(pred_s, act_s, model_name=algo)
            results.append(metrics.to_dict())

        return pd.DataFrame(results)

    def run_5_stage_ablation(
        self,
        core_algorithm: str = "LightGBM"
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Runs the 5 ablation benchmarks:
        
        1. Price-only
        2. Weather-only
        3. Price + Weather
        4. Price + Weather + Term Structure
        5. Price + Weather + Term Structure + Regime (Meta-Model)
        """
        experiment_specs = [
            ("Price-Only", "price_only"),
            ("Weather-Only", "weather_only"),
            ("Price + Weather", "price_weather"),
            ("Price + Weather + Term Structure", "price_weather_term"),
        ]

        ablation_metrics = []
        equity_curves = {}
        out_of_sample_preds = {}

        # First run experiments 1 to 4 using the chosen core algorithm
        for exp_name, feat_group in experiment_specs:
            logger.info(f"Executing {exp_name}...")
            X, y = self.dataset.get_X_y(feat_group)

            all_preds = []
            all_actuals = []
            test_dates = []

            for train_idx, test_idx in self.cv.split(X):
                X_tr, y_tr = X.iloc[train_idx].values, y.iloc[train_idx].values
                X_te, y_te = X.iloc[test_idx].values, y.iloc[test_idx].values

                preds = evaluate_model_family(X_tr, y_tr, X_te, y_te, core_algorithm)
                all_preds.extend(preds)
                all_actuals.extend(y_te)
                test_dates.extend(X.index[test_idx])

            pred_s = pd.Series(all_preds, index=test_dates)
            act_s = pd.Series(all_actuals, index=test_dates)
            out_of_sample_preds[feat_group] = pred_s

            metrics, curve_df = self.backtester.run_backtest(pred_s, act_s, model_name=exp_name)
            ablation_metrics.append(metrics.to_dict())
            equity_curves[exp_name] = curve_df

        # Now run Experiment 5: Meta-Model combining base signals + regime detection
        logger.info("Executing Experiment 5: Price + Weather + Term Structure + Regime (Meta-Model)...")
        test_idx_union = out_of_sample_preds["price_only"].index
        
        # Assemble base out-of-sample signals
        base_signals_df = pd.DataFrame({
            "price_ml": out_of_sample_preds["price_only"],
            "weather_ml": out_of_sample_preds["weather_only"],
            "term_carry": self.dataset.df.loc[test_idx_union, "term_roll_yield"],
            "momentum": self.dataset.df.loc[test_idx_union, "price_mom_21d"]
        }).dropna()

        # Regime detection on the test period
        regime_detector = RegimeDetector(n_regimes=3, method="hmm")
        regime_detector.fit(self.dataset.df.loc[test_idx_union])
        states, proba = regime_detector.predict_regimes(self.dataset.df.loc[base_signals_df.index])

        # Dynamic regime allocator
        allocator = AdaptiveSignalAllocator()
        dynamic_weights = allocator.allocate(base_signals_df, states, proba)
        meta_pred = allocator.blend_forecasts(base_signals_df, dynamic_weights)

        y_actual_meta = self.dataset.df.loc[meta_pred.index, self.dataset.target_col]
        realized_vol = self.dataset.df.loc[meta_pred.index, "price_vol_21d"]

        meta_metrics, meta_curve = self.backtester.run_backtest(
            meta_pred,
            y_actual_meta,
            model_name="+ Regime (Meta-Model)",
            realized_vol=realized_vol
        )
        ablation_metrics.append(meta_metrics.to_dict())
        equity_curves["+ Regime (Meta-Model)"] = meta_curve

        summary_df = pd.DataFrame(ablation_metrics)
        return summary_df, equity_curves

