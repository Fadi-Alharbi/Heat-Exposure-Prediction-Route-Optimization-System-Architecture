"""
Heat Exposure Prediction - Model 2 (LightGBM / Random Forest Alternative Model)
Includes Multi-Objective Weighting & Ensemble Support for ThermoRoute.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlternativeHeatExposureModel:
    """
    Model 2: Alternative ML model (RandomForest/LightGBM) 
    for heat exposure prediction and route risk scoring.
    """
    def __init__(self, model_type='rf', n_estimators=100, max_depth=10, random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        
        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'lgb':
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state
                )
            except ImportError:
                logger.warning("LightGBM not installed. Falling back to RandomForestRegressor.")
                self.model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state
                )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train Model 2 on feature matrix."""
        logger.info(f"Training Model 2 ({self.model_type.upper()})...")
        self.model.fit(X_train, y_train)
        logger.info("Model 2 training complete.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict heat exposure values."""
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model metrics."""
        preds = self.predict(X_test)
        metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "mae": mean_absolute_error(y_test, preds),
            "r2": r2_score(y_test, preds)
        }
        logger.info(f"Model 2 Evaluation Metrics: {metrics}")
        return metrics

    def save_model(self, filepath: str):
        """Save Model 2 to disk."""
        joblib.dump(self.model, filepath)
        logger.info(f"Model 2 saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load Model 2 from disk."""
        instance = cls()
        instance.model = joblib.load(filepath)
        logger.info(f"Model 2 loaded from {filepath}")
        return instance


class MultiObjectiveWeightAssignerV2:
    """
    Alternative Weight Assigner for Pathfinding in OpenStreetMap graph.
    Combines Distance, Heat Exposure Score (from Model 1/2), and Shade Penalty.
    """
    def __init__(self, alpha=0.5, beta=0.5):
        """
        alpha: Weight for distance/time
        beta: Weight for thermal exposure risk score
        """
        self.alpha = alpha
        self.beta = beta

    def compute_edge_weight(self, length: float, heat_score: float, shadow_ratio: float) -> float:
        """
        Calculates composite impedance/weight for graph edge optimization.
        Lower composite weight = Preferred route segment.
        """
        # Thermal penalty increases as shadow ratio decreases
        thermal_risk = heat_score * (1.0 - (0.5 * shadow_ratio))
        
        # Multi-objective composite cost
        cost = (self.alpha * length) + (self.beta * length * thermal_risk)
        return float(cost)
