"""
Land Surface Temperature (LST) Fetcher
───────────────────────────────────────
Placeholder module for fetching Land Surface Temperature data.

In production, this would connect to:
  - NASA MODIS (MOD11A1 / MYD11A1) via earthaccess
  - Landsat LST via Google Earth Engine

For the initial version, we provide a physics-based estimation model
that derives LST from air temperature, solar radiation, and surface type.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


@dataclass
class LSTEstimate:
    """Estimated land surface temperature for a point."""
    latitude: float
    longitude: float
    timestamp: dt.datetime
    air_temperature_c: float
    surface_type: str
    solar_radiation_wm2: float
    estimated_lst_c: float
    confidence: str  # "estimated" | "measured"


class LSTFetcher:
    """
    Estimates Land Surface Temperature when satellite data is unavailable.

    The estimation uses a simplified energy-balance approach:
        LST ≈ T_air + ΔT_surface

    where ΔT_surface depends on:
        - Solar radiation intensity
        - Surface albedo and thermal properties
        - Wind speed (convective cooling)
    """

    # Surface thermal gain coefficients (°C per 100 W/m² of solar radiation)
    SURFACE_THERMAL_GAIN = {
        "asphalt": 3.5,
        "concrete": 2.8,
        "paving_stones": 2.5,
        "compacted": 2.0,
        "gravel": 1.8,
        "sand": 2.2,
        "dirt": 1.5,
        "grass": 0.8,
        "ground": 1.5,
        "unknown": 2.0,
    }

    def estimate_lst(
        self,
        air_temperature_c: float,
        solar_radiation_wm2: float,
        surface_type: str = "asphalt",
        wind_speed_kmh: float = 10.0,
        latitude: float = 0.0,
        longitude: float = 0.0,
        timestamp: dt.datetime | None = None,
    ) -> LSTEstimate:
        """
        Estimate LST from meteorological conditions and surface type.

        Parameters
        ----------
        air_temperature_c : float
            Air temperature at 2m height (°C).
        solar_radiation_wm2 : float
            Global horizontal irradiance (W/m²).
        surface_type : str
            Road/ground surface type (e.g., "asphalt", "grass").
        wind_speed_kmh : float
            Wind speed at 10m (km/h).
        """
        ts = timestamp or dt.datetime.now(dt.timezone.utc)

        # Get surface thermal gain coefficient
        gain = self.SURFACE_THERMAL_GAIN.get(
            surface_type.lower(),
            self.SURFACE_THERMAL_GAIN["unknown"],
        )

        # Solar heating contribution
        solar_delta = gain * (solar_radiation_wm2 / 100.0)

        # Wind cooling factor (higher wind → closer LST to air temp)
        wind_ms = wind_speed_kmh / 3.6
        wind_cooling_factor = 1.0 / (1.0 + 0.15 * wind_ms)

        # Estimated LST
        delta_t = solar_delta * wind_cooling_factor
        lst = air_temperature_c + delta_t

        logger.debug(
            "LST estimate: air={air}°C, solar={sol}W/m², surface={surf} → LST={lst:.1f}°C (ΔT={dt:.1f}°C)",
            air=air_temperature_c, sol=solar_radiation_wm2,
            surf=surface_type, lst=lst, dt=delta_t,
        )

        return LSTEstimate(
            latitude=latitude,
            longitude=longitude,
            timestamp=ts,
            air_temperature_c=air_temperature_c,
            surface_type=surface_type,
            solar_radiation_wm2=solar_radiation_wm2,
            estimated_lst_c=round(lst, 2),
            confidence="estimated",
        )

    def estimate_lst_batch(
        self,
        air_temps: np.ndarray,
        solar_rads: np.ndarray,
        surface_types: list[str],
        wind_speeds: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized LST estimation for multiple segments.

        Returns an array of estimated LST values (°C).
        """
        gains = np.array([
            self.SURFACE_THERMAL_GAIN.get(s.lower(), self.SURFACE_THERMAL_GAIN["unknown"])
            for s in surface_types
        ])
        solar_deltas = gains * (solar_rads / 100.0)
        wind_ms = wind_speeds / 3.6
        wind_cooling = 1.0 / (1.0 + 0.15 * wind_ms)
        return air_temps + solar_deltas * wind_cooling
