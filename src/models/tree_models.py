"""Gradient Boosted Decision Tree Models: LightGBM and XGBoost Regressors."""

from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb


class LightGBMModel:
    """LightGBM tabular regressor optimized for non-linear weather thresholds."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        num_leaves: int = 15,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42
    ):
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=-1
        )
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_vals = X.values
        else:
            X_vals = X
            self.feature_names = [f"f_{i}" for i in range(X.shape[1])]

        y_vals = y.values if isinstance(y, pd.Series) else y
        self.model.fit(X_vals, y_vals)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_vals)

    def get_feature_importances(self) -> pd.Series:
        if hasattr(self.model, "feature_importances_") and self.feature_names:
            return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        return pd.Series()


class XGBoostModel:
    """XGBoost regressor with gradient regularization."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42
    ):
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbosity=0
        )
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            X_vals = X.values
        else:
            X_vals = X
            self.feature_names = [f"f_{i}" for i in range(X.shape[1])]

        y_vals = y.values if isinstance(y, pd.Series) else y
        self.model.fit(X_vals, y_vals)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        return self.model.predict(X_vals)

    def get_feature_importances(self) -> pd.Series:
        if hasattr(self.model, "feature_importances_") and self.feature_names:
            return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        return pd.Series()

