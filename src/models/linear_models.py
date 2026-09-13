"""Linear Regularized Models: Ridge and Lasso with Cross-Validation."""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class RegularizedLinearModel:
    """Standardized linear model with L1/L2 regularization and scaling."""

    def __init__(self, model_type: str = "ridge", cv_folds: int = 5):
        self.model_type = model_type.lower()
        self.cv_folds = cv_folds
        self.scaler = StandardScaler()
        
        if self.model_type == "lasso":
            self.regressor = LassoCV(cv=cv_folds, random_state=42, max_iter=2000)
        elif self.model_type == "elasticnet":
            self.regressor = ElasticNetCV(cv=cv_folds, l1_ratio=[0.1, 0.5, 0.9], random_state=42)
        else:
            self.regressor = RidgeCV(alphas=np.logspace(-3, 4, 20), cv=cv_folds)

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        y_vals = y.values if isinstance(y, pd.Series) else y
        
        X_scaled = self.scaler.fit_transform(X_vals)
        self.regressor.fit(X_scaled, y_vals)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_vals)
        return self.regressor.predict(X_scaled)

    def get_feature_coefficients(self, feature_names: list[str]) -> pd.Series:
        if hasattr(self.regressor, "coef_"):
            return pd.Series(self.regressor.coef_, index=feature_names)
        return pd.Series()

