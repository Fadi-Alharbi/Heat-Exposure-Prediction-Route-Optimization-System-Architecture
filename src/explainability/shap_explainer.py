"""
SHAP Explainer
──────────────
Uses SHAP (SHapley Additive exPlanations) to explain which features
drove the heat exposure prediction for a specific route or segment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed – explainability features limited")


class ShapExplainer:
    """
    Generates feature-importance explanations for heat exposure predictions.
    """

    def __init__(self, model: Any = None):
        self.model = model
        self.explainer = None

    def initialize(self, model: Any, X_background: pd.DataFrame | None = None) -> None:
        """
        Initialize the SHAP explainer with a trained model.

        Parameters
        ----------
        model : trained model object
            Must have a .predict() method.
        X_background : DataFrame, optional
            Background dataset for SHAP (subset of training data).
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available")
            return

        self.model = model
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'predict'):
                # HeatExposureModel wrapper
                underlying = model.model
            else:
                underlying = model

            if X_background is not None:
                self.explainer = shap.TreeExplainer(underlying, data=X_background)
            else:
                self.explainer = shap.TreeExplainer(underlying)
            logger.info("SHAP TreeExplainer initialized")
        except Exception as exc:
            logger.warning("Could not initialize TreeExplainer: {e}, trying KernelExplainer", e=exc)
            try:
                if X_background is not None:
                    self.explainer = shap.KernelExplainer(underlying.predict, X_background)
                    logger.info("SHAP KernelExplainer initialized")
            except Exception as exc2:
                logger.error("Could not initialize any SHAP explainer: {e}", e=exc2)

    def explain_prediction(
        self,
        features: pd.DataFrame,
    ) -> dict[str, float] | None:
        """
        Explain a single prediction using SHAP values.

        Returns a dict mapping feature_name → SHAP value (contribution).
        """
        if self.explainer is None:
            return self._fallback_importance(features)

        try:
            shap_values = self.explainer.shap_values(features)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Average across samples if multiple
            if len(shap_values.shape) > 1:
                mean_abs = np.abs(shap_values).mean(axis=0)
            else:
                mean_abs = np.abs(shap_values)

            importance = dict(zip(features.columns, mean_abs))
            # Sort by importance
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        except Exception as exc:
            logger.warning("SHAP explanation failed: {e}", e=exc)
            return self._fallback_importance(features)

    def explain_route(
        self,
        segment_features: pd.DataFrame,
    ) -> dict:
        """
        Explain why a route has a certain heat exposure level.

        Returns aggregated feature importances across all segments.
        """
        if self.explainer is None:
            return {
                "top_factors": self._fallback_importance(segment_features),
                "explanation_type": "heuristic",
            }

        try:
            shap_values = self.explainer.shap_values(segment_features)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            mean_abs = np.abs(shap_values).mean(axis=0)
            importance = dict(zip(segment_features.columns, mean_abs))
            sorted_importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )

            return {
                "top_factors": sorted_importance,
                "explanation_type": "shap",
                "n_segments": len(segment_features),
            }
        except Exception as exc:
            logger.warning("Route explanation failed: {e}", e=exc)
            return {
                "top_factors": self._fallback_importance(segment_features),
                "explanation_type": "heuristic",
            }

    @staticmethod
    def _fallback_importance(features: pd.DataFrame) -> dict[str, float]:
        """
        Heuristic feature importance when SHAP is not available.
        Based on domain knowledge of heat exposure factors.
        """
        known_importance = {
            "air_temperature_c": 0.25,
            "direct_radiation_wm2": 0.20,
            "shade_fraction": 0.15,
            "relative_humidity_pct": 0.10,
            "surface_heat_factor": 0.10,
            "lst_estimate_c": 0.08,
            "wind_speed_kmh": 0.05,
            "solar_altitude_deg": 0.04,
            "hour_of_day": 0.02,
            "diffuse_radiation_wm2": 0.01,
        }
        return {k: v for k, v in known_importance.items() if k in features.columns}
