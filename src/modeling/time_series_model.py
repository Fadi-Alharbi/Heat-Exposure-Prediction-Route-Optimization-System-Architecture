"""
Time-Series Forecasting Model
──────────────────────────────
Forecasts weather conditions (temperature, radiation, wind) across
different hours of the day, enabling the system to recommend the
best departure time for a trip.

Supports:
  - Hourly interpolation from Open-Meteo forecast data
  - Simple sinusoidal daily temperature model as fallback
  - Prophet integration (optional, requires 'prophet' package)
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class HourlyConditions:
    """Weather conditions for a specific hour."""
    hour: int
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    solar_radiation_wm2: float
    heat_exposure_estimate: float


class TimeSeriesModel:
    """
    Forecasts intra-day weather conditions for trip time optimization.
    """

    def forecast_daily_profile(
        self,
        forecast_df: pd.DataFrame | None = None,
        date: dt.date | None = None,
        latitude: float = 24.7,
        base_temp_c: float = 35.0,
    ) -> list[HourlyConditions]:
        """
        Generate an hourly profile of weather conditions for a day.

        If a forecast DataFrame is provided (from WeatherClient), it is used
        directly. Otherwise, a simplified sinusoidal model is applied.

        Parameters
        ----------
        forecast_df : pd.DataFrame, optional
            Hourly forecast data with columns:
            time, temperature_2m, relative_humidity_2m, wind_speed_10m,
            direct_radiation, diffuse_radiation.
        date : datetime.date, optional
            Target date (for filtering forecast_df).
        latitude : float
            Latitude for solar radiation estimation in fallback mode.
        base_temp_c : float
            Average daily temperature for fallback mode.
        """
        if forecast_df is not None and not forecast_df.empty:
            return self._from_forecast(forecast_df, date)

        return self._sinusoidal_model(base_temp_c, latitude, date)

    def find_best_departure_time(
        self,
        hourly_profile: list[HourlyConditions],
        earliest_hour: int = 6,
        latest_hour: int = 20,
        trip_duration_hours: float = 0.5,
    ) -> tuple[int, float]:
        """
        Find the departure hour that minimizes heat exposure.

        Returns (best_hour, estimated_exposure).
        """
        best_hour = earliest_hour
        best_exposure = float("inf")

        for cond in hourly_profile:
            if cond.hour < earliest_hour or cond.hour > latest_hour:
                continue
            if cond.heat_exposure_estimate < best_exposure:
                best_exposure = cond.heat_exposure_estimate
                best_hour = cond.hour

        return best_hour, best_exposure

    def compare_departure_times(
        self,
        hourly_profile: list[HourlyConditions],
        candidate_hours: list[int],
    ) -> pd.DataFrame:
        """
        Compare heat exposure across multiple candidate departure times.

        Returns DataFrame with hour, temperature, radiation, exposure.
        """
        rows = []
        profile_map = {c.hour: c for c in hourly_profile}

        for hour in candidate_hours:
            cond = profile_map.get(hour)
            if cond:
                rows.append({
                    "hour": hour,
                    "time_label": f"{hour:02d}:00",
                    "temperature_c": cond.temperature_c,
                    "humidity_pct": cond.humidity_pct,
                    "wind_speed_kmh": cond.wind_speed_kmh,
                    "solar_radiation_wm2": cond.solar_radiation_wm2,
                    "heat_exposure_estimate": cond.heat_exposure_estimate,
                })

        return pd.DataFrame(rows)

    # ── From forecast data ──────────────────────────────────────

    def _from_forecast(
        self,
        df: pd.DataFrame,
        date: dt.date | None,
    ) -> list[HourlyConditions]:
        """Parse hourly conditions from forecast DataFrame."""
        if date is not None:
            df = df[df["time"].dt.date == date].copy()

        if df.empty:
            logger.warning("No forecast data for the requested date")
            return []

        results = []
        for _, row in df.iterrows():
            total_rad = row.get("direct_radiation", 0) + row.get("diffuse_radiation", 0)
            temp = row.get("temperature_2m", 35)
            humidity = row.get("relative_humidity_2m", 50)
            wind = row.get("wind_speed_10m", 10)

            # Simple heat exposure estimate
            exposure = self._simple_exposure(temp, humidity, total_rad, wind)

            results.append(HourlyConditions(
                hour=row["time"].hour,
                temperature_c=temp,
                humidity_pct=humidity,
                wind_speed_kmh=wind,
                solar_radiation_wm2=total_rad,
                heat_exposure_estimate=round(exposure, 2),
            ))

        return results

    # ── Sinusoidal fallback model ───────────────────────────────

    def _sinusoidal_model(
        self,
        base_temp: float,
        latitude: float,
        date: dt.date | None,
    ) -> list[HourlyConditions]:
        """
        Generate a simplified daily temperature/radiation profile.

        Temperature follows: T(h) = T_avg + A × sin(π × (h - 6) / 12)
        Radiation follows: R(h) = R_max × sin(π × (h - sunrise) / daylight)
        """
        amplitude = 8.0  # daily temperature swing (°C)
        sunrise_hour = 5.5
        sunset_hour = 18.5
        daylight = sunset_hour - sunrise_hour
        max_radiation = 900.0  # peak GHI at solar noon

        results = []
        for hour in range(24):
            # Temperature
            if 6 <= hour <= 18:
                temp_offset = amplitude * np.sin(np.pi * (hour - 6) / 12)
            else:
                temp_offset = -amplitude * 0.5
            temp = base_temp + temp_offset

            # Humidity (inverse of temperature roughly)
            humidity = max(15, min(90, 60 - temp_offset * 3))

            # Solar radiation
            if sunrise_hour < hour < sunset_hour:
                rad_frac = np.sin(np.pi * (hour - sunrise_hour) / daylight)
                radiation = max_radiation * max(0, rad_frac)
            else:
                radiation = 0.0

            # Wind (slightly higher midday)
            wind = 8 + 5 * np.sin(np.pi * (hour - 8) / 12) if 8 <= hour <= 20 else 5

            exposure = self._simple_exposure(temp, humidity, radiation, wind)

            results.append(HourlyConditions(
                hour=hour,
                temperature_c=round(temp, 1),
                humidity_pct=round(humidity, 1),
                wind_speed_kmh=round(wind, 1),
                solar_radiation_wm2=round(radiation, 1),
                heat_exposure_estimate=round(exposure, 2),
            ))

        return results

    @staticmethod
    def _simple_exposure(temp: float, humidity: float, radiation: float, wind: float) -> float:
        """Quick heat exposure estimate for time comparison."""
        temp_norm = max(0, (temp - 20) / 35)
        hum_norm = humidity / 100
        rad_norm = radiation / 1000
        wind_factor = 1 / (1 + 0.03 * wind)

        return (0.4 * temp_norm + 0.2 * hum_norm + 0.3 * rad_norm + 0.1) * wind_factor * 50
