"""
Centralized configuration for the Heat Exposure Prediction & Route Optimization System.
Uses pydantic-settings for environment variable support.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


# ──────────────────────────────────────────────
#  Project paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # ── API Configuration ───────────────────────
    # Open-Meteo (no API key required for free tier)
    OPEN_METEO_BASE_URL: str = "https://api.open-meteo.com/v1"
    OPEN_METEO_FORECAST_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_HISTORICAL_URL: str = "https://archive-api.open-meteo.com/v1/archive"
    OPEN_METEO_ELEVATION_URL: str = "https://api.open-meteo.com/v1/elevation"

    # NASA Earthdata (optional, for MODIS LST)
    EARTHDATA_USERNAME: str = ""
    EARTHDATA_PASSWORD: str = ""

    # CDS API (optional, for ERA5)
    CDS_API_KEY: str = ""

    # ── Default Location (West Riyadh neighborhood) ─
    DEFAULT_LATITUDE: float = 24.7025
    DEFAULT_LONGITUDE: float = 46.6775
    DEFAULT_CITY: str = "Riyadh"
    DEFAULT_TIMEZONE: str = "Asia/Riyadh"

    # ── User / Trip Defaults ────────────────────
    DEFAULT_USER_SPEED_KMH: float = 15.0     # e-scooter / bicycle speed
    WALKING_SPEED_KMH: float = 5.0
    CYCLING_SPEED_KMH: float = 15.0
    DRIVING_SPEED_KMH: float = 40.0

    # ── Road Network ────────────────────────────
    NETWORK_TYPE: str = "bike"               # OSMnx network type
    SEGMENT_LENGTH_METERS: float = 50.0      # road segment granularity
    SEARCH_RADIUS_METERS: float = 5000.0     # area of interest around route

    # ── Optimization Weights ────────────────────
    ALPHA_TIME: float = Field(default=0.4, ge=0.0, le=1.0)    # weight for travel time
    BETA_HEAT: float = Field(default=0.6, ge=0.0, le=1.0)     # weight for heat exposure
    K_SHORTEST_PATHS: int = 5                # number of candidate paths

    # ── Shadow Modeling ─────────────────────────
    DEFAULT_BUILDING_HEIGHT_M: float = 9.0   # ~3 floors × 3m
    DEFAULT_TREE_HEIGHT_M: float = 6.0
    TREE_CROWN_RADIUS_M: float = 3.0

    # ── Heat Index Thresholds ───────────────────
    HEAT_EXPOSURE_LOW: float = 27.0          # °C – comfortable
    HEAT_EXPOSURE_MODERATE: float = 32.0     # °C – caution
    HEAT_EXPOSURE_HIGH: float = 39.0         # °C – danger
    HEAT_EXPOSURE_EXTREME: float = 51.0      # °C – extreme danger

    # ── Surface Thermal Properties ──────────────
    # Relative heat absorption factor (0-1) per surface type
    SURFACE_HEAT_FACTORS: dict = {
        "asphalt": 0.95,
        "concrete": 0.85,
        "paving_stones": 0.80,
        "compacted": 0.70,
        "gravel": 0.60,
        "sand": 0.55,
        "dirt": 0.50,
        "grass": 0.30,
        "ground": 0.50,
        "unknown": 0.75,
    }

    # ── ML Model ────────────────────────────────
    MODEL_NAME: str = "heat_exposure_xgb"
    MODEL_VERSION: str = "v0.1"
    RANDOM_SEED: int = 42
    TEST_SIZE: float = 0.2
    CV_FOLDS: int = 5

    # ── Server ──────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    STREAMLIT_PORT: int = 8501
    DEBUG: bool = True

    # ── Caching ─────────────────────────────────
    CACHE_DIR: str = str(PROJECT_ROOT / ".cache")
    CACHE_EXPIRE_HOURS: int = 1

    model_config = {
        "env_prefix": "HEAT_",
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton instance
settings = Settings()
