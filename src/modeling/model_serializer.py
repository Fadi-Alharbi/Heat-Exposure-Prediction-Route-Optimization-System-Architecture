"""
Model Serializer
────────────────
Save and load trained ML models with versioning and metadata.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import joblib
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import MODELS_DIR


class ModelSerializer:
    """Persists trained models to disk with version tracking."""

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model_obj: object, name: str, version: str) -> Path:
        """
        Save a model and its metadata.

        Parameters
        ----------
        model_obj : object
            The model object (HeatExposureModel instance or sklearn estimator).
        name : str
            Model name (e.g., "heat_exposure_xgb").
        version : str
            Version string (e.g., "v0.1").

        Returns
        -------
        Path to the saved model file.
        """
        filename = f"{name}_{version}.joblib"
        filepath = self.models_dir / filename

        joblib.dump(model_obj, filepath)

        # Save metadata
        meta = {
            "name": name,
            "version": version,
            "saved_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "filepath": str(filepath),
        }
        meta_path = self.models_dir / f"{name}_{version}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Model saved: {path}", path=filepath)
        return filepath

    def load(self, name: str, version: str) -> object:
        """Load a previously saved model."""
        filename = f"{name}_{version}.joblib"
        filepath = self.models_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Model not found: {filepath}")

        model = joblib.load(filepath)
        logger.info("Model loaded: {path}", path=filepath)
        return model

    def load_latest(self, name: str) -> tuple[object, str]:
        """Load the most recently saved version of a named model."""
        pattern = f"{name}_*.joblib"
        files = sorted(self.models_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        if not files:
            raise FileNotFoundError(f"No models found matching '{name}'")
        latest = files[-1]
        version = latest.stem.replace(f"{name}_", "")
        model = joblib.load(latest)
        logger.info("Loaded latest model: {path} (version: {v})", path=latest, v=version)
        return model, version

    def list_models(self) -> list[dict]:
        """List all saved models with metadata."""
        models = []
        for meta_file in self.models_dir.glob("*_meta.json"):
            with open(meta_file, "r", encoding="utf-8") as f:
                models.append(json.load(f))
        return models
