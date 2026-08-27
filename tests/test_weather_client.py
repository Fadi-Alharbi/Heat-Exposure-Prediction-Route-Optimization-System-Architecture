"""Tests for the Weather Client."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import datetime as dt
from unittest.mock import patch, MagicMock
import pytest
from src.data_ingestion.weather_client import WeatherClient, WeatherSnapshot, WeatherForecast


class TestWeatherSnapshot:
    """Test WeatherSnapshot data class."""

    def test_total_radiation(self):
        snap = WeatherSnapshot(
            timestamp=dt.datetime.now(dt.timezone.utc),
            latitude=24.7, longitude=46.7,
            temperature_c=40.0, relative_humidity_pct=30.0,
            wind_speed_kmh=15.0,
            direct_radiation_wm2=600.0, diffuse_radiation_wm2=150.0,
            cloud_cover_pct=10.0,
        )
        assert snap.total_radiation_wm2 == 750.0


class TestWeatherForecast:
    """Test WeatherForecast data class."""

    def test_at_time_returns_closest(self):
        import pandas as pd
        now = dt.datetime(2026, 8, 15, 12, 0, 0)
        df = pd.DataFrame({
            "time": [now - dt.timedelta(hours=1), now, now + dt.timedelta(hours=1)],
            "temperature_2m": [38.0, 40.0, 39.0],
            "relative_humidity_2m": [30.0, 25.0, 28.0],
            "wind_speed_10m": [10.0, 12.0, 11.0],
            "direct_radiation": [500.0, 700.0, 600.0],
            "diffuse_radiation": [100.0, 150.0, 120.0],
            "cloud_cover": [10.0, 5.0, 15.0],
            "precipitation": [0.0, 0.0, 0.0],
            "apparent_temperature": [42.0, 45.0, 43.0],
        })

        fc = WeatherForecast(latitude=24.7, longitude=46.7, timezone="Asia/Riyadh", hourly=df)
        snap = fc.at_time(now)
        assert snap is not None
        assert snap.temperature_c == 40.0

    def test_at_time_empty_returns_none(self):
        import pandas as pd
        fc = WeatherForecast(latitude=24.7, longitude=46.7, timezone="Asia/Riyadh")
        assert fc.at_time(dt.datetime.now()) is None


class TestWeatherClient:
    """Test WeatherClient (mocked HTTP)."""

    def test_client_initializes(self):
        client = WeatherClient()
        assert client is not None
