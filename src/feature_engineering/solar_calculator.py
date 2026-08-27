"""
Solar Position Calculator
─────────────────────────
Uses pvlib to compute the sun's position (altitude, azimuth) and
related quantities for any location and time. This is essential for:
  - Shadow estimation (shadow length = height / tan(altitude))
  - Direct Normal Irradiance decomposition
  - Determining sunrise/sunset windows for trip planning
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

try:
    import pvlib
    from pvlib.location import Location
    PVLIB_AVAILABLE = True
except ImportError:
    PVLIB_AVAILABLE = False
    logger.warning("pvlib not installed – using simplified solar calculations")


@dataclass
class SolarPosition:
    """Sun position at a given time and location."""
    timestamp: dt.datetime
    latitude: float
    longitude: float
    altitude_deg: float     # elevation angle above horizon (0-90)
    azimuth_deg: float      # compass direction (0=N, 90=E, 180=S, 270=W)
    zenith_deg: float       # angle from vertical (= 90 - altitude)
    is_daytime: bool
    sunrise: dt.datetime | None = None
    sunset: dt.datetime | None = None

    @property
    def altitude_rad(self) -> float:
        return np.radians(self.altitude_deg)

    @property
    def azimuth_rad(self) -> float:
        return np.radians(self.azimuth_deg)


class SolarCalculator:
    """Computes solar geometry using pvlib or a simplified fallback."""

    def get_solar_position(
        self,
        latitude: float,
        longitude: float,
        timestamp: dt.datetime,
        altitude_m: float = 0.0,
    ) -> SolarPosition:
        """
        Calculate the sun's position for a given location and time.

        Parameters
        ----------
        latitude : float
            Degrees north (-90 to 90).
        longitude : float
            Degrees east (-180 to 180).
        timestamp : datetime
            Must be timezone-aware (UTC preferred).
        altitude_m : float
            Site elevation in meters above sea level.
        """
        if PVLIB_AVAILABLE:
            return self._pvlib_position(latitude, longitude, timestamp, altitude_m)
        return self._simplified_position(latitude, longitude, timestamp)

    def get_solar_positions_hourly(
        self,
        latitude: float,
        longitude: float,
        date: dt.date,
        timezone: str = "UTC",
        altitude_m: float = 0.0,
    ) -> pd.DataFrame:
        """
        Get hourly solar positions for an entire day.

        Returns DataFrame with columns: time, altitude, azimuth, zenith, is_daytime.
        """
        import pytz
        tz = pytz.timezone(timezone) if isinstance(timezone, str) else timezone
        times = pd.date_range(
            start=dt.datetime.combine(date, dt.time(0, 0), tzinfo=tz),
            end=dt.datetime.combine(date, dt.time(23, 0), tzinfo=tz),
            freq="h",
        )

        rows = []
        for t in times:
            sp = self.get_solar_position(latitude, longitude, t, altitude_m)
            rows.append({
                "time": t,
                "altitude_deg": sp.altitude_deg,
                "azimuth_deg": sp.azimuth_deg,
                "zenith_deg": sp.zenith_deg,
                "is_daytime": sp.is_daytime,
            })

        return pd.DataFrame(rows)

    # ── pvlib implementation ────────────────────────────────────

    def _pvlib_position(
        self,
        latitude: float,
        longitude: float,
        timestamp: dt.datetime,
        altitude_m: float,
    ) -> SolarPosition:
        loc = Location(latitude, longitude, altitude=altitude_m)
        times = pd.DatetimeIndex([timestamp])
        sp = loc.get_solarposition(times)

        altitude = float(sp["apparent_elevation"].iloc[0])
        azimuth = float(sp["azimuth"].iloc[0])
        zenith = float(sp["apparent_zenith"].iloc[0])

        # Sunrise / sunset
        sunrise, sunset = None, None
        try:
            date = timestamp.date()
            sun_times = loc.get_sun_rise_set_transit(
                pd.DatetimeIndex([dt.datetime.combine(date, dt.time(12, 0), tzinfo=timestamp.tzinfo)])
            )
            sunrise = sun_times["sunrise"].iloc[0]
            sunset = sun_times["sunset"].iloc[0]
            if hasattr(sunrise, 'to_pydatetime'):
                sunrise = sunrise.to_pydatetime()
            if hasattr(sunset, 'to_pydatetime'):
                sunset = sunset.to_pydatetime()
        except Exception:
            pass

        return SolarPosition(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude_deg=max(altitude, 0.0),
            azimuth_deg=azimuth,
            zenith_deg=zenith,
            is_daytime=altitude > 0,
            sunrise=sunrise,
            sunset=sunset,
        )

    # ── Simplified fallback (no pvlib) ──────────────────────────

    @staticmethod
    def _simplified_position(
        latitude: float,
        longitude: float,
        timestamp: dt.datetime,
    ) -> SolarPosition:
        """
        Simplified solar position using basic astronomical formulas.
        Accurate to within ~1° for most applications.
        """
        import math

        # Day of year
        doy = timestamp.timetuple().tm_yday
        hour = timestamp.hour + timestamp.minute / 60.0

        # Solar declination (Spencer, 1971)
        B = math.radians((360 / 365) * (doy - 81))
        declination = math.radians(23.45 * math.sin(B))

        # Hour angle
        solar_noon = 12.0  # simplified
        hour_angle = math.radians(15 * (hour - solar_noon))

        # Solar altitude
        lat_rad = math.radians(latitude)
        sin_alt = (math.sin(lat_rad) * math.sin(declination)
                   + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle))
        altitude_rad = math.asin(max(-1, min(1, sin_alt)))
        altitude_deg = math.degrees(altitude_rad)

        # Solar azimuth
        cos_az = ((math.sin(declination) - math.sin(lat_rad) * sin_alt)
                  / (math.cos(lat_rad) * math.cos(altitude_rad) + 1e-10))
        cos_az = max(-1, min(1, cos_az))
        azimuth_deg = math.degrees(math.acos(cos_az))
        if hour_angle > 0:
            azimuth_deg = 360 - azimuth_deg

        return SolarPosition(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude_deg=max(altitude_deg, 0.0),
            azimuth_deg=azimuth_deg,
            zenith_deg=90.0 - max(altitude_deg, 0.0),
            is_daytime=altitude_deg > 0,
        )
