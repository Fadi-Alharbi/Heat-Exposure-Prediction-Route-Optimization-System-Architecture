"""
Model Trainer
─────────────
End-to-end pipeline for training, evaluating, and cross-validating
the heat exposure prediction model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings
from src.modeling.heat_exposure_model import HeatExposureModel, ModelConfig, FEATURE_COLUMNS
from src.modeling.model_serializer import ModelSerializer


@dataclass
class TrainingResult:
    """Results from a model training run."""
    train_mae: float
    train_rmse: float
    train_r2: float
    test_mae: float
    test_rmse: float
    test_r2: float
    cv_mae_mean: float | None = None
    cv_mae_std: float | None = None
    cv_rmse_mean: float | None = None
    cv_rmse_std: float | None = None
    feature_importance: pd.Series | None = None


class ModelTrainer:
    """
    Orchestrates the training pipeline for heat exposure models.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.serializer = ModelSerializer()

    def train_and_evaluate(
        self,
        data: pd.DataFrame,
        target_column: str = "heat_exposure_score",
        test_size: float | None = None,
        run_cv: bool = True,
        save_model: bool = True,
    ) -> tuple[HeatExposureModel, TrainingResult]:
        """
        Full training pipeline: split → train → evaluate → CV → save.

        Parameters
        ----------
        data : pd.DataFrame
            Training data with features and target column.
        target_column : str
            Name of the target variable column.
        test_size : float
            Fraction of data for testing.
        run_cv : bool
            Whether to run k-fold cross-validation.
        save_model : bool
            Whether to save the trained model to disk.

        Returns
        -------
        (HeatExposureModel, TrainingResult)
        """
        ts = test_size or settings.TEST_SIZE

        # Validate data
        available_features = [c for c in FEATURE_COLUMNS if c in data.columns]
        if len(available_features) < 3:
            raise ValueError(
                f"Insufficient features. Need at least 3, found: {available_features}"
            )
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not in data")

        X = data[available_features].fillna(0.0)
        y = data[target_column]

        logger.info(
            "Training with {n} samples, {f} features, target='{t}'",
            n=len(X), f=len(available_features), t=target_column,
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ts, random_state=settings.RANDOM_SEED,
        )

        # Train model
        model = HeatExposureModel(self.config)
        train_metrics = model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred_test = model.predict(X_test)
        test_metrics = {
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_r2": r2_score(y_test, y_pred_test),
        }

        logger.info(
            "Test metrics – MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}",
            mae=test_metrics["test_mae"],
            rmse=test_metrics["test_rmse"],
            r2=test_metrics["test_r2"],
        )

        # Cross-validation
        cv_results = {}
        if run_cv:
            cv_results = self._cross_validate(X, y)

        # Feature importance
        importance = model.feature_importance()

        result = TrainingResult(
            train_mae=train_metrics["mae"],
            train_rmse=train_metrics["rmse"],
            train_r2=train_metrics["r2"],
            test_mae=test_metrics["test_mae"],
            test_rmse=test_metrics["test_rmse"],
            test_r2=test_metrics["test_r2"],
            cv_mae_mean=cv_results.get("cv_mae_mean"),
            cv_mae_std=cv_results.get("cv_mae_std"),
            cv_rmse_mean=cv_results.get("cv_rmse_mean"),
            cv_rmse_std=cv_results.get("cv_rmse_std"),
            feature_importance=importance,
        )

        # Save model
        if save_model:
            self.serializer.save(model, settings.MODEL_NAME, settings.MODEL_VERSION)

        return model, result

    def _cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Run k-fold cross-validation."""
        logger.info("Running {k}-fold cross-validation …", k=settings.CV_FOLDS)

        model = HeatExposureModel(self.config)
        kf = KFold(n_splits=settings.CV_FOLDS, shuffle=True, random_state=settings.RANDOM_SEED)

        mae_scores = []
        rmse_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = HeatExposureModel(self.config)
            fold_model.fit(X_train, y_train)
            y_pred = fold_model.predict(X_val)

            mae_scores.append(mean_absolute_error(y_val, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        results = {
            "cv_mae_mean": np.mean(mae_scores),
            "cv_mae_std": np.std(mae_scores),
            "cv_rmse_mean": np.mean(rmse_scores),
            "cv_rmse_std": np.std(rmse_scores),
        }

        logger.info(
            "CV results – MAE: {mae:.3f} ± {mae_std:.3f}, RMSE: {rmse:.3f} ± {rmse_std:.3f}",
            mae=results["cv_mae_mean"], mae_std=results["cv_mae_std"],
            rmse=results["cv_rmse_mean"], rmse_std=results["cv_rmse_std"],
        )

        return results

    @staticmethod
    def generate_synthetic_training_data(
        n_samples: int = 5000,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Generate synthetic training data for initial model development.

        Uses physics-based formulas to create realistic
        feature-target relationships.
        """
        rng = np.random.RandomState(random_seed)

        hours = rng.randint(0, 24, n_samples)
        temps = 25 + 15 * np.sin(np.pi * (hours - 6) / 12) + rng.normal(0, 2, n_samples)
        temps = np.clip(temps, 15, 55)

        humidity = 40 - 15 * np.sin(np.pi * (hours - 6) / 12) + rng.normal(0, 8, n_samples)
        humidity = np.clip(humidity, 10, 95)

        wind = 5 + 5 * rng.random(n_samples)
        direct_rad = np.where(
            (hours >= 6) & (hours <= 18),
            700 * np.sin(np.pi * (hours - 6) / 12) * (0.5 + 0.5 * rng.random(n_samples)),
            0,
        )
        diffuse_rad = direct_rad * (0.15 + 0.1 * rng.random(n_samples))

        shade = rng.random(n_samples) * 0.8
        surface = 0.3 + rng.random(n_samples) * 0.65
        solar_alt = np.where(
            (hours >= 6) & (hours <= 18),
            70 * np.sin(np.pi * (hours - 6) / 12),
            0,
        )
        lst = temps + (direct_rad / 100) * surface * (1 - shade) + rng.normal(0, 1, n_samples)

        # Target: physics-based heat exposure score
        exposure = (
            0.35 * np.clip((temps - 20) / 35, 0, 1)
            + 0.15 * humidity / 100
            + 0.20 * np.clip((direct_rad + diffuse_rad) / 1000, 0, 1) * (1 - shade)
            + 0.15 * surface
            + 0.15 * np.clip((lst - 20) / 40, 0, 1)
        ) / (1 + 0.03 * wind) * 55 + rng.normal(0, 1.5, n_samples)

        df = pd.DataFrame({
            "air_temperature_c": temps.round(1),
            "relative_humidity_pct": humidity.round(1),
            "wind_speed_kmh": wind.round(1),
            "direct_radiation_wm2": direct_rad.round(1),
            "diffuse_radiation_wm2": diffuse_rad.round(1),
            "shade_fraction": shade.round(3),
            "surface_heat_factor": surface.round(3),
            "solar_altitude_deg": solar_alt.round(1),
            "hour_of_day": hours,
            "lst_estimate_c": lst.round(1),
            "heat_exposure_score": exposure.round(2),
        })

        logger.info("Generated {n} synthetic training samples", n=n_samples)
        return df
