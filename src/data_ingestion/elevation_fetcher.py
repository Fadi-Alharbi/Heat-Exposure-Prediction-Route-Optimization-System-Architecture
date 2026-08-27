"""
Elevation Fetcher
─────────────────
Fetches terrain elevation data from the Open-Meteo Elevation API.
Used to account for slope and altitude in heat exposure calculations.
"""

from __future__ import annotations

import requests
from loguru import logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


class ElevationFetcher:
    """Fetches elevation data from the Open-Meteo Elevation API."""

    def __init__(self) -> None:
        self._session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503])
        self._session.mount("https://", HTTPAdapter(max_retries=retries))

    def get_elevation(self, latitude: float, longitude: float) -> float | None:
        """
        Get elevation (meters above sea level) for a single point.

        Returns None on failure.
        """
        try:
            resp = self._session.get(
                settings.OPEN_METEO_ELEVATION_URL,
                params={"latitude": latitude, "longitude": longitude},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            elevations = data.get("elevation", [])
            if elevations:
                elev = elevations[0]
                logger.debug("Elevation at ({lat}, {lon}): {e}m",
                             lat=latitude, lon=longitude, e=elev)
                return float(elev)
        except Exception as exc:
            logger.warning("Elevation fetch failed for ({lat}, {lon}): {e}",
                           lat=latitude, lon=longitude, e=exc)
        return None

    def get_elevations_batch(
        self,
        latitudes: list[float],
        longitudes: list[float],
    ) -> list[float | None]:
        """
        Get elevations for multiple points in a single API call.

        The Open-Meteo Elevation API accepts comma-separated coordinates.
        """
        if not latitudes or not longitudes:
            return []

        # API supports multiple coordinates in one request
        try:
            lat_str = ",".join(str(x) for x in latitudes)
            lon_str = ",".join(str(x) for x in longitudes)
            resp = self._session.get(
                settings.OPEN_METEO_ELEVATION_URL,
                params={"latitude": lat_str, "longitude": lon_str},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            elevations = data.get("elevation", [])
            logger.info("Fetched {n} elevations in batch", n=len(elevations))
            return [float(e) if e is not None else None for e in elevations]
        except Exception as exc:
            logger.warning("Batch elevation fetch failed: {e}", e=exc)
            return [None] * len(latitudes)

    def get_slope_between(
        self,
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """
        Calculate slope (in degrees) between two points.

        Returns 0.0 if elevation data is unavailable.
        """
        import math
        elevs = self.get_elevations_batch([lat1, lat2], [lon1, lon2])
        if elevs[0] is None or elevs[1] is None:
            return 0.0

        elev_diff = elevs[1] - elevs[0]

        # Haversine horizontal distance
        R = 6_371_000  # Earth radius in meters
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1))
             * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        horiz_dist = 2 * R * math.asin(math.sqrt(a))

        if horiz_dist < 1.0:
            return 0.0

        slope_rad = math.atan2(elev_diff, horiz_dist)
        return math.degrees(slope_rad)
