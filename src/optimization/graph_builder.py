"""
Graph Builder
─────────────
Constructs a weighted NetworkX graph from the road network,
enriched with heat exposure costs per edge.
"""

from __future__ import annotations

import datetime as dt

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

from src.data_ingestion.weather_client import WeatherClient, WeatherSnapshot
from src.data_ingestion.lst_fetcher import LSTFetcher
from src.feature_engineering.solar_calculator import SolarCalculator
from src.feature_engineering.shadow_estimator import ShadowEstimator
from src.feature_engineering.surface_classifier import SurfaceClassifier
from src.feature_engineering.heat_index_calculator import HeatIndexCalculator
from src.modeling.heat_exposure_model import HeatExposureModel

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


class GraphBuilder:
    """
    Enriches a road network graph with heat exposure weights.

    For each edge, calculates:
    - travel_time_s: time to traverse at given speed
    - heat_exposure_score: predicted heat stress level
    - combined_cost: α × time + β × heat_exposure
    """

    def __init__(self):
        self.solar_calc = SolarCalculator()
        self.shadow_est = ShadowEstimator()
        self.surface_cls = SurfaceClassifier()
        self.heat_calc = HeatIndexCalculator()
        self.lst_fetcher = LSTFetcher()
        self.heat_model = HeatExposureModel()

    def build_weighted_graph(
        self,
        graph: nx.MultiDiGraph,
        weather: WeatherSnapshot,
        trip_time: dt.datetime,
        user_speed_kmh: float | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        buildings_gdf=None,
        trees_gdf=None,
    ) -> nx.MultiDiGraph:
        """
        Add heat exposure weights to all edges in the graph.

        Parameters
        ----------
        graph : nx.MultiDiGraph
            OSMnx road network.
        weather : WeatherSnapshot
            Current/forecast weather conditions.
        trip_time : datetime
            When the trip will take place.
        user_speed_kmh : float
            User travel speed.
        alpha : float
            Weight for time component (0-1).
        beta : float
            Weight for heat exposure component (0-1).
        buildings_gdf : GeoDataFrame, optional
            Building footprints with heights.
        trees_gdf : GeoDataFrame, optional
            Tree locations with heights.
        """
        speed = user_speed_kmh or settings.DEFAULT_USER_SPEED_KMH
        a = alpha if alpha is not None else settings.ALPHA_TIME
        b = beta if beta is not None else settings.BETA_HEAT

        # Ensure weights sum approximately to 1
        total = a + b
        if total > 0:
            a, b = a / total, b / total

        # Get solar position
        solar = self.solar_calc.get_solar_position(
            weather.latitude, weather.longitude, trip_time,
        )

        logger.info(
            "Building weighted graph: {n} edges, solar alt={alt:.1f}°, T={t}°C",
            n=graph.number_of_edges(), alt=solar.altitude_deg, t=weather.temperature_c,
        )

        speed_ms = speed * 1000 / 3600  # km/h → m/s

        for u, v, key, data in graph.edges(keys=True, data=True):
            edge_length = data.get("length", 0.0)
            if edge_length < 0.1:
                continue

            # Travel time
            travel_time_s = edge_length / speed_ms

            # Surface classification
            surface_props = self.surface_cls.classify_edge(data)

            # Shade estimation
            u_data = graph.nodes[u]
            v_data = graph.nodes[v]
            mid_lat = (u_data["y"] + v_data["y"]) / 2
            mid_lon = (u_data["x"] + v_data["x"]) / 2

            # Simplified bearing
            import math
            dlat = v_data["y"] - u_data["y"]
            dlon = v_data["x"] - u_data["x"]
            bearing = math.degrees(math.atan2(dlon, dlat)) % 360

            shade_result = self.shadow_est.estimate_shade_fraction(
                segment_center_lat=mid_lat,
                segment_center_lon=mid_lon,
                segment_bearing_deg=bearing,
                segment_width_m=6.0,
                solar_position=solar,
                nearby_buildings=self._gdf_to_nearby_list(buildings_gdf, mid_lat, mid_lon, 100),
                nearby_trees=self._gdf_to_nearby_list(trees_gdf, mid_lat, mid_lon, 50),
            )

            # LST estimation
            lst = self.lst_fetcher.estimate_lst(
                air_temperature_c=weather.temperature_c,
                solar_radiation_wm2=weather.total_radiation_wm2,
                surface_type=surface_props.surface_type,
                wind_speed_kmh=weather.wind_speed_kmh,
            )

            # Build feature vector for heat model
            features = pd.DataFrame([{
                "air_temperature_c": weather.temperature_c,
                "relative_humidity_pct": weather.relative_humidity_pct,
                "wind_speed_kmh": weather.wind_speed_kmh,
                "direct_radiation_wm2": weather.direct_radiation_wm2,
                "diffuse_radiation_wm2": weather.diffuse_radiation_wm2,
                "shade_fraction": shade_result.shade_fraction,
                "surface_heat_factor": surface_props.heat_factor,
                "solar_altitude_deg": solar.altitude_deg,
                "hour_of_day": trip_time.hour,
                "lst_estimate_c": lst.estimated_lst_c,
            }])

            heat_score = float(self.heat_model.predict(features)[0])

            # Cumulative heat exposure = score × duration (in minutes)
            duration_min = travel_time_s / 60.0
            cumulative_heat = heat_score * duration_min

            # Combined cost
            combined_cost = a * travel_time_s + b * cumulative_heat

            # Store on edge
            data["travel_time_s"] = travel_time_s
            data["heat_exposure_score"] = heat_score
            data["cumulative_heat_exposure"] = cumulative_heat
            data["shade_fraction"] = shade_result.shade_fraction
            data["surface_type"] = surface_props.surface_type
            data["surface_heat_factor"] = surface_props.heat_factor
            data["combined_cost"] = combined_cost

        logger.info("Graph weighted successfully (α={a:.2f}, β={b:.2f})", a=a, b=b)
        return graph

    @staticmethod
    def _gdf_to_nearby_list(gdf, lat, lon, radius_m) -> list[dict] | None:
        """Convert GeoDataFrame to list of nearby objects for shadow estimation."""
        if gdf is None or gdf.empty:
            return None

        try:
            from shapely.geometry import Point
            center = Point(lon, lat)
            # Rough degree-to-meter conversion
            deg_radius = radius_m / 111_320

            nearby = []
            for _, row in gdf.iterrows():
                centroid = row.geometry.centroid
                if abs(centroid.x - lon) < deg_radius and abs(centroid.y - lat) < deg_radius:
                    nearby.append({
                        "lat": centroid.y,
                        "lon": centroid.x,
                        "height_m": row.get("estimated_height_m", 9.0),
                        "footprint_radius_m": 10.0,
                        "crown_radius_m": 3.0,
                    })
                    if len(nearby) >= 20:  # limit for performance
                        break
            return nearby if nearby else None
        except Exception:
            return None
