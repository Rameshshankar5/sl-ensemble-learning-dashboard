import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor


class RFANNHybridRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid Tabular Architecture: Random Forest + Artificial Neural Network (MLP).
    - Stage 1 (Random Forest): Captures non-linear decision boundaries & feature interactions.
    - Stage 2 (ANN): Trained on out-of-fold RF residuals to model smooth continuous depreciation trends.
    """

    def __init__(self, rf_estimators=200, ann_layers=(128, 64), random_state=42):
        self.rf_estimators = rf_estimators
        self.ann_layers = ann_layers
        self.random_state = random_state

    def fit(self, X, y):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        # Generate out-of-fold predictions to prevent residual data leakage
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        oof_rf = np.zeros(len(y_arr))

        for train_idx, val_idx in kf.split(X_arr):
            rf_fold = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
            rf_fold.fit(X_arr[train_idx], y_arr[train_idx])
            oof_rf[val_idx] = rf_fold.predict(X_arr[val_idx])

        residuals = y_arr - oof_rf

        # Fit full Random Forest on target and ANN on residuals
        self.rf_ = RandomForestRegressor(
            n_estimators=self.rf_estimators, random_state=self.random_state, n_jobs=-1
        )
        self.rf_.fit(X_arr, y_arr)

        self.ann_ = MLPRegressor(
            hidden_layer_sizes=self.ann_layers,
            activation="relu",
            max_iter=600,
            early_stopping=True,
            random_state=self.random_state,
        )
        self.ann_.fit(X_arr, residuals)
        return self

    def predict(self, X):
        X_arr = np.asarray(X)
        return self.rf_.predict(X_arr) + self.ann_.predict(X_arr)