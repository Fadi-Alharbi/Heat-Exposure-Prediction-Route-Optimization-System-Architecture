"""
Shadow Estimator (Simplified 2.5D Model)
─────────────────────────────────────────
Estimates the shadow fraction for road segments based on nearby
buildings and trees, using a simplified 2.5D projection.

Key formula:
    shadow_length = object_height / tan(solar_altitude)
    shadow_direction = solar_azimuth + 180°  (opposite to sun)

When no building/tree data is available, shade_fraction defaults to 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from loguru import logger
from pyproj import Transformer
from shapely import affinity
from shapely.ops import transform, unary_union

from src.feature_engineering.solar_calculator import SolarPosition


@dataclass
class ShadowResult:
    """Shadow analysis result for a road segment."""
    segment_id: str | int
    shade_fraction: float      # 0.0 (fully exposed) to 1.0 (fully shaded)
    shadow_sources: int        # number of objects casting shadows on this segment
    dominant_source: str       # "building" | "tree" | "none"


class ShadowEstimator:
    """
    Estimates shade coverage on road segments from nearby structures.

    Uses a simplified 2.5D approach:
    1. For each building/tree near the segment, project its shadow
       based on solar altitude and azimuth.
    2. Check if the shadow footprint intersects the road segment.
    3. Aggregate shade fraction.
    """

    def __init__(self, max_shadow_distance_m: float = 100.0):
        self.max_shadow_distance_m = max_shadow_distance_m
        # Riyadh is in UTM zone 38N.  Shadow geometry must be calculated in
        # metres, never directly in latitude/longitude degrees.
        self._to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32638", always_xy=True)

    def estimate_shade_fraction(
        self,
        segment_center_lat: float,
        segment_center_lon: float,
        segment_bearing_deg: float,
        segment_width_m: float,
        solar_position: SolarPosition,
        nearby_buildings: list[dict] | None = None,
        nearby_trees: list[dict] | None = None,
    ) -> ShadowResult:
        """
        Estimate the shade fraction for a road segment.

        Parameters
        ----------
        segment_center_lat, segment_center_lon : float
            Center coordinates of the road segment.
        segment_bearing_deg : float
            Compass bearing of the road (0-360°).
        segment_width_m : float
            Estimated road width in meters.
        solar_position : SolarPosition
            Current sun position.
        nearby_buildings : list of dict, optional
            Each dict has keys: lat, lon, height_m, footprint_radius_m.
        nearby_trees : list of dict, optional
            Each dict has keys: lat, lon, height_m, crown_radius_m.

        Returns
        -------
        ShadowResult
        """
        if not solar_position.is_daytime or solar_position.altitude_deg < 2.0:
            # Night or very low sun → no meaningful shadow analysis
            return ShadowResult(
                segment_id=0,
                shade_fraction=1.0 if not solar_position.is_daytime else 0.0,
                shadow_sources=0,
                dominant_source="none",
            )

        buildings = nearby_buildings or []
        trees = nearby_trees or []

        if not buildings and not trees:
            return ShadowResult(
                segment_id=0,
                shade_fraction=0.0,
                shadow_sources=0,
                dominant_source="none",
            )

        shade_contributions = []
        sources = {"building": 0, "tree": 0}

        # Process buildings
        for bldg in buildings:
            contrib = self._shadow_contribution(
                obj_lat=bldg["lat"],
                obj_lon=bldg["lon"],
                obj_height=bldg["height_m"],
                obj_radius=bldg.get("footprint_radius_m", 10.0),
                seg_lat=segment_center_lat,
                seg_lon=segment_center_lon,
                seg_bearing=segment_bearing_deg,
                seg_width=segment_width_m,
                solar=solar_position,
            )
            if contrib > 0:
                shade_contributions.append(contrib)
                sources["building"] += 1

        # Process trees
        for tree in trees:
            contrib = self._shadow_contribution(
                obj_lat=tree["lat"],
                obj_lon=tree["lon"],
                obj_height=tree["height_m"],
                obj_radius=tree.get("crown_radius_m", 3.0),
                seg_lat=segment_center_lat,
                seg_lon=segment_center_lon,
                seg_bearing=segment_bearing_deg,
                seg_width=segment_width_m,
                solar=solar_position,
            )
            if contrib > 0:
                # Trees provide partial shade (dappled light)
                shade_contributions.append(contrib * 0.7)
                sources["tree"] += 1

        # Combine shade (cap at 1.0)
        if shade_contributions:
            # Use probabilistic combination: 1 - product(1 - shade_i)
            total_shade = 1.0 - np.prod([1.0 - s for s in shade_contributions])
            total_shade = min(total_shade, 1.0)
        else:
            total_shade = 0.0

        dominant = max(sources, key=sources.get) if any(sources.values()) else "none"

        return ShadowResult(
            segment_id=0,
            shade_fraction=round(total_shade, 3),
            shadow_sources=sources["building"] + sources["tree"],
            dominant_source=dominant,
        )

    def estimate_shade_from_footprints(
        self,
        road_geometry,
        road_width_m: float,
        solar_position: SolarPosition,
        buildings: list[dict],
    ) -> ShadowResult:
        """Measure shade using projected building footprints and road area.

        Each footprint is extruded opposite the sun by ``height / tan(altitude)``.
        The returned fraction is the shadowed road-surface area, rather than a
        proximity score based on the building centroid.
        """
        if not solar_position.is_daytime:
            return ShadowResult(0, 1.0, 0, "none")
        if solar_position.altitude_deg < 2.0 or not buildings:
            return ShadowResult(0, 0.0, 0, "none")

        road_m = transform(self._to_metric.transform, road_geometry)
        road_area = road_m.buffer(max(road_width_m, 1.0) / 2.0, cap_style=2)
        if road_area.is_empty or road_area.area == 0:
            return ShadowResult(0, 0.0, 0, "none")

        shadow_direction = math.radians((solar_position.azimuth_deg + 180.0) % 360.0)
        intersections = []
        source_count = 0
        for building in buildings:
            footprint = building.get("geometry")
            if footprint is None or footprint.is_empty:
                continue
            height = float(building.get("height_m", 9.0))
            shadow_length = min(
                self.max_shadow_distance_m,
                height / math.tan(solar_position.altitude_rad),
            )
            footprint_m = transform(self._to_metric.transform, footprint)
            dx = shadow_length * math.sin(shadow_direction)
            dy = shadow_length * math.cos(shadow_direction)
            # Convex hull of footprint and its translated copy is the 2.5D
            # ground-shadow footprint for a vertical building approximation.
            shadow = unary_union([
                footprint_m,
                affinity.translate(footprint_m, xoff=dx, yoff=dy),
            ]).convex_hull
            overlap = shadow.intersection(road_area)
            if not overlap.is_empty:
                intersections.append(overlap)
                source_count += 1

        if not intersections:
            return ShadowResult(0, 0.0, 0, "none")
        shaded_area = unary_union(intersections).area
        return ShadowResult(
            segment_id=0,
            shade_fraction=round(min(1.0, shaded_area / road_area.area), 3),
            shadow_sources=source_count,
            dominant_source="building",
        )

    def _shadow_contribution(
        self,
        obj_lat: float, obj_lon: float,
        obj_height: float, obj_radius: float,
        seg_lat: float, seg_lon: float,
        seg_bearing: float, seg_width: float,
        solar: SolarPosition,
    ) -> float:
        """
        Calculate how much shade one object casts on a road segment.

        Returns a value between 0.0 and 1.0.
        """
        # Shadow length
        tan_alt = math.tan(solar.altitude_rad)
        if tan_alt < 0.01:
            return 0.0
        shadow_length = obj_height / tan_alt

        if shadow_length > self.max_shadow_distance_m:
            shadow_length = self.max_shadow_distance_m

        # Shadow direction (opposite to sun)
        shadow_dir_deg = (solar.azimuth_deg + 180.0) % 360.0

        # Distance from object to segment center (approximate)
        dist = self._haversine_m(obj_lat, obj_lon, seg_lat, seg_lon)

        # Bearing from object to segment center
        bearing = self._bearing_deg(obj_lat, obj_lon, seg_lat, seg_lon)

        # Angular difference between shadow direction and bearing to segment
        angle_diff = abs(shadow_dir_deg - bearing)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        # Shadow reaches the segment if:
        # 1. The distance is within shadow length + object radius
        # 2. The angular alignment is reasonable
        effective_reach = shadow_length + obj_radius
        if dist > effective_reach:
            return 0.0

        # Angular coverage (wider objects cast wider shadows)
        angular_width = math.degrees(math.atan2(obj_radius, max(dist, 1.0)))

        if angle_diff > angular_width + 30:
            return 0.0

        # Shade intensity diminishes with distance and angle
        dist_factor = max(0.0, 1.0 - (dist / effective_reach))
        angle_factor = max(0.0, 1.0 - (angle_diff / (angular_width + 30)))

        return dist_factor * angle_factor

    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Approximate distance in meters between two coordinates."""
        R = 6_371_000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
             * math.sin(dlon / 2) ** 2)
        return 2 * R * math.asin(math.sqrt(a))

    @staticmethod
    def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Compass bearing from point 1 to point 2."""
        dlon = math.radians(lon2 - lon1)
        lat1r, lat2r = math.radians(lat1), math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2r)
        y = (math.cos(lat1r) * math.sin(lat2r)
             - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360
