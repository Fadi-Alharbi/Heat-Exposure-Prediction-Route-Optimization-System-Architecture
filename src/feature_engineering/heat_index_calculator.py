"""
Heat Index Calculator
─────────────────────
Computes multiple thermal comfort / heat stress indices:
  - Heat Index (NWS / Rothfusz regression)
  - Simplified WBGT (Wet Bulb Globe Temperature)
  - Simplified UTCI (Universal Thermal Climate Index)
  - Apparent Temperature

These are used as features and/or labels for the ML model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class ThermalIndices:
    """Collection of thermal comfort indices for a point/time."""
    heat_index_c: float
    wbgt_c: float
    apparent_temperature_c: float
    thermal_stress_category: str   # "comfortable" | "caution" | "danger" | "extreme"


class HeatIndexCalculator:
    """
    Calculates multiple heat stress indices from basic meteorological inputs.
    """

    # ── Heat Index (NWS Rothfusz Regression) ────────────────────

    @staticmethod
    def heat_index(temperature_c: float, relative_humidity: float) -> float:
        """
        Calculate Heat Index using the NWS Rothfusz regression equation.

        Parameters
        ----------
        temperature_c : float
            Air temperature in °C.
        relative_humidity : float
            Relative humidity in % (0-100).

        Returns
        -------
        float
            Heat Index in °C.
        """
        # Convert to Fahrenheit for NWS formula
        T = temperature_c * 9.0 / 5.0 + 32.0
        RH = relative_humidity

        # Simple formula for low temperatures
        if T < 80:
            hi_f = 0.5 * (T + 61.0 + (T - 68.0) * 1.2 + RH * 0.094)
            return (hi_f - 32.0) * 5.0 / 9.0

        # Full Rothfusz regression
        hi_f = (
            -42.379
            + 2.04901523 * T
            + 10.14333127 * RH
            - 0.22475541 * T * RH
            - 6.83783e-3 * T**2
            - 5.481717e-2 * RH**2
            + 1.22874e-3 * T**2 * RH
            + 8.5282e-4 * T * RH**2
            - 1.99e-6 * T**2 * RH**2
        )

        # Adjustments
        if RH < 13 and 80 <= T <= 112:
            adj = -((13 - RH) / 4) * math.sqrt((17 - abs(T - 95)) / 17)
            hi_f += adj
        elif RH > 85 and 80 <= T <= 87:
            adj = ((RH - 85) / 10) * ((87 - T) / 5)
            hi_f += adj

        return (hi_f - 32.0) * 5.0 / 9.0

    # ── WBGT (Simplified) ──────────────────────────────────────

    @staticmethod
    def wbgt_simplified(
        temperature_c: float,
        relative_humidity: float,
        solar_radiation_wm2: float = 0.0,
        wind_speed_ms: float = 1.0,
    ) -> float:
        """
        Simplified Wet Bulb Globe Temperature estimation.

        Uses the Liljegren-inspired approximation:
            WBGT ≈ 0.7 × Tw + 0.2 × Tg + 0.1 × Ta

        Where:
            Tw (wet bulb) is estimated from T and RH
            Tg (globe temperature) is estimated from T and solar radiation
            Ta = air temperature
        """
        # Estimate wet-bulb temperature (Stull, 2011 approximation)
        T = temperature_c
        RH = relative_humidity
        Tw = T * math.atan(0.151977 * math.sqrt(RH + 8.313659)) \
             + math.atan(T + RH) \
             - math.atan(RH - 1.676331) \
             + 0.00391838 * RH**1.5 * math.atan(0.023101 * RH) \
             - 4.686035

        # Estimate globe temperature
        # Globe temp rises with solar radiation and decreases with wind
        solar_factor = 0.01 * solar_radiation_wm2  # simplified
        wind_cooling = 0.5 * math.sqrt(max(wind_speed_ms, 0.1))
        Tg = T + solar_factor - wind_cooling

        # WBGT formula
        wbgt = 0.7 * Tw + 0.2 * Tg + 0.1 * T
        return round(wbgt, 2)

    # ── Apparent Temperature ───────────────────────────────────

    @staticmethod
    def apparent_temperature(
        temperature_c: float,
        relative_humidity: float,
        wind_speed_ms: float = 1.0,
    ) -> float:
        """
        Australian Bureau of Meteorology apparent temperature formula.

        AT = Ta + 0.348 × e − 0.70 × ws − 0.21 × ws − 4.25

        where e = (RH/100) × 6.105 × exp(17.27 × Ta / (237.7 + Ta))
        """
        T = temperature_c
        RH = relative_humidity
        ws = wind_speed_ms

        # Water vapour pressure (hPa)
        e = (RH / 100.0) * 6.105 * math.exp(17.27 * T / (237.7 + T))

        at = T + 0.348 * e - 0.70 * ws - 4.25
        return round(at, 2)

    # ── Combined calculation ───────────────────────────────────

    def calculate_all(
        self,
        temperature_c: float,
        relative_humidity: float,
        wind_speed_kmh: float = 10.0,
        solar_radiation_wm2: float = 0.0,
    ) -> ThermalIndices:
        """
        Calculate all thermal indices and determine stress category.
        """
        wind_ms = wind_speed_kmh / 3.6

        hi = self.heat_index(temperature_c, relative_humidity)
        wbgt = self.wbgt_simplified(
            temperature_c, relative_humidity,
            solar_radiation_wm2, wind_ms,
        )
        at = self.apparent_temperature(temperature_c, relative_humidity, wind_ms)

        # Categorize based on WBGT thresholds
        if wbgt < 27:
            category = "comfortable"
        elif wbgt < 32:
            category = "caution"
        elif wbgt < 39:
            category = "danger"
        else:
            category = "extreme"

        return ThermalIndices(
            heat_index_c=round(hi, 2),
            wbgt_c=wbgt,
            apparent_temperature_c=at,
            thermal_stress_category=category,
        )

    # ── Vectorized batch calculation ───────────────────────────

    @staticmethod
    def heat_index_batch(
        temperatures: np.ndarray,
        humidities: np.ndarray,
    ) -> np.ndarray:
        """Vectorized Heat Index for arrays of T and RH."""
        T_f = temperatures * 9.0 / 5.0 + 32.0
        RH = humidities

        # Simple formula
        hi_simple = 0.5 * (T_f + 61.0 + (T_f - 68.0) * 1.2 + RH * 0.094)

        # Full Rothfusz
        hi_full = (
            -42.379
            + 2.04901523 * T_f
            + 10.14333127 * RH
            - 0.22475541 * T_f * RH
            - 6.83783e-3 * T_f**2
            - 5.481717e-2 * RH**2
            + 1.22874e-3 * T_f**2 * RH
            + 8.5282e-4 * T_f * RH**2
            - 1.99e-6 * T_f**2 * RH**2
        )

        # Use simple formula where T < 80°F
        hi_f = np.where(T_f < 80, hi_simple, hi_full)

        return (hi_f - 32.0) * 5.0 / 9.0
