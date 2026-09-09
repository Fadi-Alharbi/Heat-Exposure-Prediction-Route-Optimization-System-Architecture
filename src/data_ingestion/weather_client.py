"""
Open-Meteo Weather Client
─────────────────────────
Fetches current, forecast, and historical weather data from the Open-Meteo API.
Free tier – no API key required.

Data retrieved:
  - Air temperature (°C)
  - Relative humidity (%)
  - Wind speed (km/h)
  - Direct & diffuse solar radiation (W/m²)
  - Cloud cover (%)
  - Precipitation (mm)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


# ──────────────────────────────────────────────────────────────────────
#  Data containers
# ──────────────────────────────────────────────────────────────────────

@dataclass
class WeatherSnapshot:
    """Weather conditions at a specific time and location."""
    timestamp: dt.datetime
    latitude: float
    longitude: float
    temperature_c: float
    relative_humidity_pct: float
    wind_speed_kmh: float
    direct_radiation_wm2: float
    diffuse_radiation_wm2: float
    cloud_cover_pct: float
    precipitation_mm: float = 0.0
    apparent_temperature_c: float | None = None

    @property
    def total_radiation_wm2(self) -> float:
        """Global Horizontal Irradiance (GHI) approximation."""
        return self.direct_radiation_wm2 + self.diffuse_radiation_wm2


@dataclass
class WeatherForecast:
    """Hourly forecast for a location."""
    latitude: float
    longitude: float
    timezone: str
    hourly: pd.DataFrame = field(default_factory=pd.DataFrame)

    def at_time(self, target_time: dt.datetime) -> WeatherSnapshot | None:
        """Return the closest hourly snapshot to *target_time*."""
        if self.hourly.empty:
            return None
            
        # Ensure target_time is timezone-naive to match Open-Meteo's local time formatting
        if target_time.tzinfo is not None:
            import pytz
            try:
                tz = pytz.timezone(self.timezone)
                target_time = pd.Timestamp(target_time).tz_convert(tz).tz_localize(None)
            except Exception:
                target_time = target_time.replace(tzinfo=None) # Fallback

        idx = (self.hourly["time"] - target_time).abs().idxmin()
        row = self.hourly.iloc[idx]
        return WeatherSnapshot(
            timestamp=row["time"],
            latitude=self.latitude,
            longitude=self.longitude,
            temperature_c=row["temperature_2m"],
            relative_humidity_pct=row["relative_humidity_2m"],
            wind_speed_kmh=row["wind_speed_10m"],
            direct_radiation_wm2=row["direct_radiation"],
            diffuse_radiation_wm2=row["diffuse_radiation"],
            cloud_cover_pct=row["cloud_cover"],
            precipitation_mm=row.get("precipitation", 0.0),
            apparent_temperature_c=row.get("apparent_temperature"),
        )


# ──────────────────────────────────────────────────────────────────────
#  Client
# ──────────────────────────────────────────────────────────────────────

class WeatherClient:
    """Thin wrapper around the Open-Meteo REST API."""

    # Hourly variables we always request
    HOURLY_VARS = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "direct_radiation",
        "diffuse_radiation",
        "cloud_cover",
        "precipitation",
        "apparent_temperature",
    ]

    def __init__(self) -> None:
        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503])
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    # ── Public API ──────────────────────────────────────────────────

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 3,
        timezone: str | None = None,
    ) -> WeatherForecast:
        """Fetch hourly forecast for the next *days* (max 16)."""
        tz = timezone or settings.DEFAULT_TIMEZONE
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(self.HOURLY_VARS),
            "timezone": tz,
            "forecast_days": min(days, 16),
        }
        data = self._get(settings.OPEN_METEO_FORECAST_URL, params)
        df = self._parse_hourly(data)
        logger.info(
            "Fetched {n}-hour forecast for ({lat}, {lon})",
            n=len(df), lat=latitude, lon=longitude,
        )
        return WeatherForecast(
            latitude=data.get("latitude", latitude),
            longitude=data.get("longitude", longitude),
            timezone=tz,
            hourly=df,
        )

    def get_historical(
        self,
        latitude: float,
        longitude: float,
        start_date: dt.date,
        end_date: dt.date,
        timezone: str | None = None,
    ) -> WeatherForecast:
        """Fetch hourly historical weather (ERA5-backed) between two dates."""
        tz = timezone or settings.DEFAULT_TIMEZONE
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": ",".join(self.HOURLY_VARS),
            "timezone": tz,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }
        data = self._get(settings.OPEN_METEO_HISTORICAL_URL, params)
        df = self._parse_hourly(data)
        logger.info(
            "Fetched historical weather {s} → {e} for ({lat}, {lon}): {n} rows",
            s=start_date, e=end_date, lat=latitude, lon=longitude, n=len(df),
        )
        return WeatherForecast(
            latitude=data.get("latitude", latitude),
            longitude=data.get("longitude", longitude),
            timezone=tz,
            hourly=df,
        )

    def get_current(
        self,
        latitude: float,
        longitude: float,
        timezone: str | None = None,
    ) -> WeatherSnapshot | None:
        """Convenience: fetch forecast and return the closest-to-now snapshot."""
        forecast = self.get_forecast(latitude, longitude, days=1, timezone=timezone)
        now = dt.datetime.now(dt.timezone.utc)
        return forecast.at_time(now)

    # ── Internals ───────────────────────────────────────────────────

    def _get(self, url: str, params: dict[str, Any]) -> dict:
        resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_hourly(data: dict) -> pd.DataFrame:
        hourly = data.get("hourly", {})
        if not hourly:
            return pd.DataFrame()
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        return df
