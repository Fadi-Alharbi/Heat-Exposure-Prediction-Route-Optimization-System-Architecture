"""
Heat Exposure Prediction Model
───────────────────────────────
XGBoost / RandomForest regression model that predicts a heat exposure
score for each road segment given meteorological and spatial features.

Features (input):
  - air_temperature_c
  - relative_humidity_pct
  - wind_speed_kmh
  - direct_radiation_wm2
  - diffuse_radiation_wm2
  - shade_fraction (0-1)
  - surface_heat_factor (0-1)
  - solar_altitude_deg
  - hour_of_day (0-23)
  - lst_estimate_c
  - road_bearing_deg

Target (output):
  - heat_exposure_score (continuous, higher = worse)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost not installed – using RandomForest only")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


# Feature columns expected by the model
FEATURE_COLUMNS = [
    "air_temperature_c",
    "relative_humidity_pct",
    "wind_speed_kmh",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "shade_fraction",
    "surface_heat_factor",
    "solar_altitude_deg",
    "hour_of_day",
    "lst_estimate_c",
]


@dataclass
class ModelConfig:
    """Configuration for the heat exposure model."""
    model_type: Literal["xgboost", "random_forest"] = "xgboost"
    n_estimators: int = 200
    max_depth: int = 8
    learning_rate: float = 0.1
    min_child_weight: int = 5
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = settings.RANDOM_SEED
    n_jobs: int = -1


class HeatExposureModel:
    """
    Predicts per-segment heat exposure score using an ensemble model.

    Can operate in two modes:
    1. Trained mode: uses a fitted ML model
    2. Heuristic mode: uses a physics-based formula (no training needed)
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model: Any = None
        self.scaler: StandardScaler | None = None
        self.is_trained = False
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the underlying sklearn/xgboost estimator."""
        if self.config.model_type == "xgboost" and XGB_AVAILABLE:
            self.model = XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_weight=self.config.min_child_weight,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                objective="reg:squarederror",
            )
            logger.info("Initialized XGBoost regressor")
        else:
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
            logger.info("Initialized RandomForest regressor")

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> dict[str, float]:
        """
        Train the model on labeled data.

        Returns training metrics (MAE, RMSE, R²).
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Select only expected features
        X_clean = self._prepare_features(X)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_clean)

        # Train
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Training metrics
        y_pred = self.model.predict(X_scaled)
        metrics = {
            "mae": mean_absolute_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "r2": r2_score(y, y_pred),
        }
        logger.info("Model trained – MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}", **metrics)
        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict heat exposure scores.

        If not trained, falls back to heuristic mode.
        """
        if not self.is_trained:
            logger.debug("Using heuristic heat exposure model (not trained)")
            return self.heuristic_predict(X)

        X_clean = self._prepare_features(X)
        X_scaled = self.scaler.transform(X_clean)
        return self.model.predict(X_scaled)

    def heuristic_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Physics-based heuristic for heat exposure estimation.

        Formula (weighted combination):
            score = (0.35 × temp_norm + 0.15 × humidity_norm + 0.20 × radiation_norm
                     + 0.15 × surface_factor + 0.15 × (1 - shade_fraction))
                    × wind_reduction_factor

        Where each component is normalized to [0, 1].
        """
        df = X.copy()

        # Normalize temperature: 20-55°C → 0-1
        temp_norm = np.clip((df.get("air_temperature_c", 35) - 20) / 35, 0, 1)

        # Normalize humidity: 0-100% → 0-1
        humidity_norm = np.clip(df.get("relative_humidity_pct", 50) / 100, 0, 1)

        # Normalize radiation: 0-1000 W/m² → 0-1
        total_rad = (df.get("direct_radiation_wm2", 0) + df.get("diffuse_radiation_wm2", 0))
        rad_norm = np.clip(total_rad / 1000, 0, 1)

        # Surface heat factor (already 0-1)
        surface = df.get("surface_heat_factor", 0.75)

        # Shade effect (1 - shade = exposure)
        shade = df.get("shade_fraction", 0)
        exposure = 1.0 - shade

        # Wind cooling factor
        wind = df.get("wind_speed_kmh", 10)
        wind_factor = 1.0 / (1.0 + 0.03 * wind)

        # Combine
        score = (
            0.35 * temp_norm
            + 0.15 * humidity_norm
            + 0.20 * rad_norm
            + 0.15 * surface
            + 0.15 * exposure
        ) * wind_factor

        # Scale to meaningful range (0-50+, mimicking WBGT-like scale)
        return np.array(score * 55)

    def feature_importance(self) -> pd.Series | None:
        """Return feature importances if model is trained."""
        if not self.is_trained:
            return None
        importances = self.model.feature_importances_
        return pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and validate feature columns."""
        available = [c for c in FEATURE_COLUMNS if c in X.columns]
        missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if missing:
            logger.warning("Missing features (using defaults): {m}", m=missing)
            for col in missing:
                X[col] = 0.0
        return X[FEATURE_COLUMNS].fillna(0.0)
