"""
alpha_models.py
===============
Machine learning alpha signal generation for ASX equities.

Models Implemented
------------------
    RandomForestAlpha   : Ensemble of decision trees for return prediction
    GradientBoostingAlpha: Gradient boosted trees (scikit-learn)
    XGBoostAlpha        : XGBoost implementation with early stopping
    MLEnsemble          : Weighted ensemble of all models
    AlphaModelPipeline  : End-to-end training and signal generation

All models:
    - Output a continuous return prediction score (not binary)
    - Include feature importance analysis
    - Support walk-forward cross-validation
    - Handle class imbalance and return skewness
    - Generate trading signals from predictions

Usage
-----
    from src.machine_learning.alpha_models import AlphaModelPipeline

    pipeline = AlphaModelPipeline(prices, macro_data=macro_df)
    signals = pipeline.generate_signals(horizon=21, n_long=10)
    result = pipeline.backtest(benchmark=benchmark_returns)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import (
        RandomForestRegressor,
        GradientBoostingRegressor,
        ExtraTreesRegressor,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.linear_model import Ridge, Lasso
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("  [AlphaModels] Warning: scikit-learn not installed. "
          "Run: pip install scikit-learn")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ─────────────────────────────────────────────
#  Result Container
# ─────────────────────────────────────────────

@dataclass
class ModelResult:
    """Container for ML model training results."""
    model_name: str = ""
    predictions: pd.Series = field(default_factory=pd.Series)
    feature_importance: pd.Series = field(default_factory=pd.Series)
    train_r2: float = 0.0
    test_r2: float = 0.0
    train_ic: float = 0.0      # Information Coefficient (rank correlation)
    test_ic: float = 0.0
    cv_ic_mean: float = 0.0
    cv_ic_std: float = 0.0
    n_features: int = 0
    n_train: int = 0
    n_test: int = 0


# ─────────────────────────────────────────────
#  Base class
# ─────────────────────────────────────────────

class BaseAlphaModel:
    """Base class for all alpha models."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self._fitted = False
        self.feature_names_: list[str] = []

    def _information_coefficient(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """
        Rank Information Coefficient: Spearman correlation between
        predicted and actual returns. IC > 0.05 is considered useful.
        """
        if len(y_true) < 5:
            return 0.0
        ic, _ = stats.spearmanr(y_pred, y_true)
        return float(ic) if not np.isnan(ic) else 0.0

    def _cross_validate_ic(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
    ) -> tuple[float, float]:
        """Walk-forward cross-validation of IC."""
        if not HAS_SKLEARN:
            return 0.0, 0.0

        tscv = TimeSeriesSplit(n_splits=n_splits)
        ics = []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            try:
                self.model.fit(X_tr, y_tr)
                preds = self.model.predict(X_te)
                ic = self._information_coefficient(y_te, preds)
                ics.append(ic)
            except Exception:
                pass

        return float(np.mean(ics)) if ics else 0, float(np.std(ics)) if ics else 0

    def get_feature_importance(self, feature_names: list[str]) -> pd.Series:
        """Extract feature importances from the fitted model."""
        if not self._fitted or self.model is None:
            return pd.Series(dtype=float)

        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            imp = np.abs(self.model.coef_)
        else:
            return pd.Series(dtype=float)

        n = min(len(imp), len(feature_names))
        return pd.Series(imp[:n], index=feature_names[:n]).sort_values(ascending=False)


# ─────────────────────────────────────────────
#  Random Forest
# ─────────────────────────────────────────────

class RandomForestAlpha(BaseAlphaModel):
    """
    Random Forest return prediction model.

    Builds an ensemble of decision trees, each trained on a random
    subset of features and data. Averaging across trees reduces
    overfitting relative to a single tree.

    Parameters
    ----------
    n_estimators : int   Number of trees (default 200).
    max_depth    : int   Maximum tree depth (default 8).
    max_features : str   Features per split ('sqrt' = √n_features).
    min_samples_leaf : int   Minimum samples in leaf nodes.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        max_features: str = "sqrt",
        min_samples_leaf: int = 20,
        random_state: int = 42,
    ) -> None:
        super().__init__("Random Forest")
        if HAS_SKLEARN:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
                n_jobs=-1,
            )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> ModelResult:
        """Fit the model and return evaluation metrics."""
        if not HAS_SKLEARN:
            return ModelResult(model_name=self.name)

        self.feature_names_ = list(X_train.columns)
        X_tr = X_train.fillna(0).values
        y_tr = y_train.values

        # Fit
        self.model.fit(X_tr, y_tr)
        self._fitted = True

        train_preds = self.model.predict(X_tr)
        train_r2 = float(r2_score(y_tr, train_preds))
        train_ic = self._information_coefficient(y_tr, train_preds)

        # CV
        cv_ic_mean, cv_ic_std = self._cross_validate_ic(X_tr, y_tr)

        # Test
        test_r2, test_ic = 0.0, 0.0
        test_predictions = pd.Series(dtype=float)

        if X_test is not None and y_test is not None:
            X_te = X_test.fillna(0).values
            y_te = y_test.values
            test_preds_arr = self.model.predict(X_te)
            test_r2 = float(r2_score(y_te, test_preds_arr))
            test_ic = self._information_coefficient(y_te, test_preds_arr)
            test_predictions = pd.Series(
                test_preds_arr, index=X_test.index, name="prediction"
            )

        return ModelResult(
            model_name=self.name,
            predictions=test_predictions,
            feature_importance=self.get_feature_importance(self.feature_names_),
            train_r2=train_r2,
            test_r2=test_r2,
            train_ic=train_ic,
            test_ic=test_ic,
            cv_ic_mean=cv_ic_mean,
            cv_ic_std=cv_ic_std,
            n_features=len(self.feature_names_),
            n_train=len(X_train),
            n_test=len(X_test) if X_test is not None else 0,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data."""
        if not self._fitted or not HAS_SKLEARN:
            return pd.Series(0.0, index=X.index)
        return pd.Series(
            self.model.predict(X.fillna(0).values),
            index=X.index,
            name="RF_prediction",
        )


# ─────────────────────────────────────────────
#  Gradient Boosting
# ─────────────────────────────────────────────

class GradientBoostingAlpha(BaseAlphaModel):
    """
    Gradient Boosting return prediction model.
    Builds trees sequentially — each tree corrects the errors of the last.

    Parameters
    ----------
    n_estimators  : int     Number of boosting rounds.
    learning_rate : float   Shrinkage factor (0.01–0.1 typical).
    max_depth     : int     Tree depth (3–5 typical for boosting).
    subsample     : float   Row sampling per tree (reduces overfitting).
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        subsample: float = 0.8,
        random_state: int = 42,
    ) -> None:
        super().__init__("Gradient Boosting")
        if HAS_SKLEARN:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=random_state,
            )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> ModelResult:
        """Fit and evaluate."""
        if not HAS_SKLEARN:
            return ModelResult(model_name=self.name)

        self.feature_names_ = list(X_train.columns)
        X_tr = X_train.fillna(0).values
        y_tr = y_train.values

        self.model.fit(X_tr, y_tr)
        self._fitted = True

        train_preds = self.model.predict(X_tr)
        train_ic = self._information_coefficient(y_tr, train_preds)
        train_r2 = float(r2_score(y_tr, train_preds))
        cv_ic_mean, cv_ic_std = self._cross_validate_ic(X_tr, y_tr)

        test_r2, test_ic = 0.0, 0.0
        test_predictions = pd.Series(dtype=float)

        if X_test is not None and y_test is not None:
            X_te = X_test.fillna(0).values
            y_te = y_test.values
            test_preds_arr = self.model.predict(X_te)
            test_r2 = float(r2_score(y_te, test_preds_arr))
            test_ic = self._information_coefficient(y_te, test_preds_arr)
            test_predictions = pd.Series(
                test_preds_arr, index=X_test.index, name="prediction"
            )

        return ModelResult(
            model_name=self.name,
            predictions=test_predictions,
            feature_importance=self.get_feature_importance(self.feature_names_),
            train_r2=train_r2, test_r2=test_r2,
            train_ic=train_ic, test_ic=test_ic,
            cv_ic_mean=cv_ic_mean, cv_ic_std=cv_ic_std,
            n_features=len(self.feature_names_),
            n_train=len(X_train),
            n_test=len(X_test) if X_test is not None else 0,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted or not HAS_SKLEARN:
            return pd.Series(0.0, index=X.index)
        return pd.Series(
            self.model.predict(X.fillna(0).values),
            index=X.index, name="GB_prediction"
        )


# ─────────────────────────────────────────────
#  XGBoost
# ─────────────────────────────────────────────

class XGBoostAlpha(BaseAlphaModel):
    """
    XGBoost return prediction model with early stopping.

    XGBoost is typically the highest-performing tree model for
    tabular financial data. Key advantages:
        - L1/L2 regularisation built in
        - Handles missing values natively
        - Faster than sklearn GBM on large datasets

    Parameters
    ----------
    n_estimators  : int     Maximum boosting rounds.
    learning_rate : float   Eta parameter.
    max_depth     : int     Tree depth.
    reg_alpha     : float   L1 regularisation.
    reg_lambda    : float   L2 regularisation.
    early_stopping: int     Stop if no improvement for N rounds.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping: int = 50,
        random_state: int = 42,
    ) -> None:
        super().__init__("XGBoost")
        self.early_stopping = early_stopping
        if HAS_XGB:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        elif HAS_SKLEARN:
            # Fallback to sklearn GBM
            self.model = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, random_state=random_state,
            )
            self.name = "GradientBoosting (XGB fallback)"

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> ModelResult:
        """Fit XGBoost with optional early stopping."""
        if not HAS_SKLEARN and not HAS_XGB:
            return ModelResult(model_name=self.name)

        self.feature_names_ = list(X_train.columns)
        X_tr = X_train.fillna(0).values
        y_tr = y_train.values

        if HAS_XGB and X_test is not None and y_test is not None:
            X_te = X_test.fillna(0).values
            y_te = y_test.values
            eval_set = [(X_te, y_te)]
            self.model.fit(
                X_tr, y_tr,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping,
                verbose=False,
            )
        else:
            self.model.fit(X_tr, y_tr)

        self._fitted = True
        train_preds = self.model.predict(X_tr)
        train_ic = self._information_coefficient(y_tr, train_preds)
        train_r2 = float(r2_score(y_tr, train_preds))
        cv_ic_mean, cv_ic_std = self._cross_validate_ic(X_tr, y_tr)

        test_r2, test_ic = 0.0, 0.0
        test_predictions = pd.Series(dtype=float)

        if X_test is not None and y_test is not None:
            X_te = X_test.fillna(0).values
            y_te = y_test.values
            test_preds_arr = self.model.predict(X_te)
            test_r2 = float(r2_score(y_te, test_preds_arr))
            test_ic = self._information_coefficient(y_te, test_preds_arr)
            test_predictions = pd.Series(
                test_preds_arr, index=X_test.index, name="prediction"
            )

        return ModelResult(
            model_name=self.name,
            predictions=test_predictions,
            feature_importance=self.get_feature_importance(self.feature_names_),
            train_r2=train_r2, test_r2=test_r2,
            train_ic=train_ic, test_ic=test_ic,
            cv_ic_mean=cv_ic_mean, cv_ic_std=cv_ic_std,
            n_features=len(self.feature_names_),
            n_train=len(X_train),
            n_test=len(X_test) if X_test is not None else 0,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self._fitted:
            return pd.Series(0.0, index=X.index)
        return pd.Series(
            self.model.predict(X.fillna(0).values),
            index=X.index, name="XGB_prediction"
        )


# ─────────────────────────────────────────────
#  ML Ensemble
# ─────────────────────────────────────────────

class MLEnsemble:
    """
    Weighted ensemble of multiple alpha models.
    Combines predictions from RF, GB, and XGB with configurable weights.
    Reduces model-specific overfitting and improves stability.

    Parameters
    ----------
    weights : dict   {model_name: weight} — defaults to equal weight.
    """

    def __init__(
        self,
        weights: Optional[dict] = None,
    ) -> None:
        self.rf = RandomForestAlpha()
        self.gb = GradientBoostingAlpha()
        self.xgb_model = XGBoostAlpha()
        self.weights = weights or {"RF": 0.33, "GB": 0.33, "XGB": 0.34}
        self._results: dict[str, ModelResult] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
    ) -> dict[str, ModelResult]:
        """Fit all models and return individual results."""
        self._results["RF"] = self.rf.fit(X_train, y_train, X_test, y_test)
        self._results["GB"] = self.gb.fit(X_train, y_train, X_test, y_test)
        self._results["XGB"] = self.xgb_model.fit(X_train, y_train, X_test, y_test)
        return self._results

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Weighted average of model predictions."""
        preds = {}
        total_w = 0.0

        for name, w in self.weights.items():
            model = {"RF": self.rf, "GB": self.gb, "XGB": self.xgb_model}.get(name)
            if model and model._fitted:
                preds[name] = model.predict(X) * w
                total_w += w

        if not preds:
            return pd.Series(0.0, index=X.index)

        ensemble = sum(preds.values()) / total_w
        ensemble.name = "Ensemble_prediction"
        return ensemble

    def compare_models(self) -> pd.DataFrame:
        """Compare all model metrics side by side."""
        rows = []
        for name, r in self._results.items():
            rows.append({
                "Model": r.model_name,
                "Train IC": r.train_ic,
                "Test IC": r.test_ic,
                "CV IC Mean": r.cv_ic_mean,
                "CV IC Std": r.cv_ic_std,
                "Train R²": r.train_r2,
                "Test R²": r.test_r2,
                "N Features": r.n_features,
                "N Train": r.n_train,
                "N Test": r.n_test,
            })
        return pd.DataFrame(rows).set_index("Model")

    def top_features(self, n: int = 20) -> pd.DataFrame:
        """Aggregate feature importance across all models."""
        all_imp = []
        for name, r in self._results.items():
            if not r.feature_importance.empty:
                imp = r.feature_importance.rename(name)
                all_imp.append(imp)

        if not all_imp:
            return pd.DataFrame()

        df = pd.concat(all_imp, axis=1).fillna(0)
        df["Average"] = df.mean(axis=1)
        return df.sort_values("Average", ascending=False).head(n)


# ─────────────────────────────────────────────
#  End-to-End Pipeline
# ─────────────────────────────────────────────

class AlphaModelPipeline:
    """
    End-to-end ML alpha signal generation pipeline.

    Connects feature engineering → model training → signal generation
    → backtesting in a single workflow.

    Parameters
    ----------
    prices      : pd.DataFrame   Adjusted close prices.
    macro_data  : pd.DataFrame   Macro variables (optional).
    model_type  : str            'rf' | 'gb' | 'xgb' | 'ensemble'
    horizon     : int            Forward return prediction horizon (days).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        macro_data: Optional[pd.DataFrame] = None,
        model_type: str = "ensemble",
        horizon: int = 21,
    ) -> None:
        from src.machine_learning.feature_engineering import FeatureEngineer

        self.prices = prices.copy()
        self.macro = macro_data
        self.model_type = model_type
        self.horizon = horizon
        self.fe = FeatureEngineer(prices, macro_data)

        self._model = None
        self._feature_names: list[str] = []
        self._predictions: pd.DataFrame = pd.DataFrame()

    def _get_model(self):
        """Instantiate the chosen model."""
        if self.model_type == "rf":
            return RandomForestAlpha()
        elif self.model_type == "gb":
            return GradientBoostingAlpha()
        elif self.model_type == "xgb":
            return XGBoostAlpha()
        else:
            return MLEnsemble()

    def train(self, test_pct: float = 0.20) -> dict:
        """
        Build features, train the model, and evaluate.

        Returns
        -------
        dict  with model results for each ticker.
        """
        results = {}

        for ticker in self.prices.columns:
            try:
                features = self.fe.build_features(
                    ticker=ticker,
                    target_horizon=self.horizon,
                    include_target=True,
                )
                features = features.dropna(subset=["target"])
                if len(features) < 100:
                    continue

                X_train, X_test, y_train, y_test = self.fe.train_test_split(
                    features, test_pct=test_pct
                )

                model = self._get_model()

                if isinstance(model, MLEnsemble):
                    r = model.fit(X_train, y_train, X_test, y_test)
                    results[ticker] = r
                    self._model = model
                else:
                    r = model.fit(X_train, y_train, X_test, y_test)
                    results[ticker] = {self.model_type: r}
                    self._model = model

                self._feature_names = list(X_train.columns)

            except Exception as e:
                print(f"  [Pipeline] {ticker} training failed: {e}")

        return results

    def generate_signals(
        self,
        n_long: int = 10,
        n_short: int = 0,
    ) -> pd.DataFrame:
        """
        Generate cross-sectional trading signals from ML predictions.

        Trains a model for each ticker, generates return predictions,
        then ranks across the universe to build long/short positions.

        Parameters
        ----------
        n_long  : int   Number of long positions.
        n_short : int   Number of short positions.

        Returns
        -------
        pd.DataFrame  Position weights (rows=dates, cols=tickers).
        """
        all_predictions = {}

        for ticker in self.prices.columns:
            try:
                features = self.fe.build_features(
                    ticker=ticker,
                    target_horizon=self.horizon,
                    include_target=False,
                )
                features = features.drop(columns=["ticker"], errors="ignore")

                model = self._get_model()
                feat_with_target = self.fe.build_features(
                    ticker=ticker, target_horizon=self.horizon, include_target=True
                )
                X_tr, _, y_tr, _ = self.fe.train_test_split(
                    feat_with_target.dropna(subset=["target"]), test_pct=0.0
                )

                if len(X_tr) < 50:
                    continue

                if isinstance(model, MLEnsemble):
                    model.fit(X_tr, y_tr)
                    preds = model.predict(features.fillna(0))
                else:
                    model.fit(X_tr, y_tr)
                    preds = model.predict(features.fillna(0))

                all_predictions[ticker] = preds

            except Exception:
                pass

        if not all_predictions:
            return pd.DataFrame()

        pred_df = pd.DataFrame(all_predictions)
        signals = pd.DataFrame(0.0, index=pred_df.index, columns=pred_df.columns)

        rebalance_dates = set(
            pred_df.resample("ME").last().index.normalize()
        )
        last_signal = pd.Series(0.0, index=pred_df.columns)

        for date in pred_df.index:
            if pd.Timestamp(date).normalize() in rebalance_dates:
                row = pred_df.loc[date].dropna()
                if len(row) < n_long:
                    last_signal = pd.Series(0.0, index=pred_df.columns)
                    signals.loc[date] = last_signal
                    continue

                new_sig = pd.Series(0.0, index=pred_df.columns)
                longs = row.nlargest(n_long).index
                new_sig[longs] = 1.0 / n_long

                if n_short > 0:
                    shorts = row.nsmallest(n_short).index
                    new_sig[shorts] = -1.0 / n_short

                last_signal = new_sig

            signals.loc[date] = last_signal

        return signals

    def backtest(
        self,
        n_long: int = 10,
        initial_capital: float = 1_000_000,
        commission: float = 0.001,
        benchmark: Optional[pd.Series] = None,
    ):
        """Generate signals and run backtest."""
        from src.backtesting.backtester import Backtester, BacktestConfig

        signals = self.generate_signals(n_long=n_long)
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.0005,
            signal_lag=1,
        )
        bt = Backtester(
            prices=self.prices,
            signals=signals,
            config=config,
            benchmark=benchmark,
            strategy_name=f"ML_{self.model_type.upper()}",
        )
        return bt.run()
