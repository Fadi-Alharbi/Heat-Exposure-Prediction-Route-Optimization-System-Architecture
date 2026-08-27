"""
OpenStreetMap Data Fetcher
──────────────────────────
Uses OSMnx to download road networks, buildings, and tree data
for a given location, then converts them to analysis-ready structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from loguru import logger
from shapely.geometry import Point

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


@dataclass
class UrbanData:
    """Container for all spatial data fetched from OSM."""
    road_graph: nx.MultiDiGraph | None = None
    buildings_gdf: gpd.GeoDataFrame | None = None
    trees_gdf: gpd.GeoDataFrame | None = None
    landuse_gdf: gpd.GeoDataFrame | None = None
    center: tuple[float, float] = (0.0, 0.0)
    radius_m: float = 0.0


class OSMFetcher:
    """Fetches and preprocesses OpenStreetMap data via OSMnx."""

    def fetch_area(
        self,
        latitude: float,
        longitude: float,
        radius_m: float | None = None,
        network_type: str | None = None,
    ) -> UrbanData:
        """
        Download road network, buildings, trees, and land use around a point.

        Parameters
        ----------
        latitude, longitude : float
            Center point coordinates.
        radius_m : float, optional
            Search radius in meters (default from settings).
        network_type : str, optional
            OSMnx network type: 'walk', 'bike', 'drive', 'all'.
        """
        rad = radius_m or settings.SEARCH_RADIUS_METERS
        net_type = network_type or settings.NETWORK_TYPE

        data = UrbanData(center=(latitude, longitude), radius_m=rad)
        center_point = (latitude, longitude)

        # ── Road network ────────────────────────────────────────
        logger.info("Fetching road network ({type}) within {r}m of ({lat}, {lon})",
                     type=net_type, r=rad, lat=latitude, lon=longitude)
        try:
            G = ox.graph_from_point(center_point, dist=rad, network_type=net_type)
            # Add edge lengths if not present
            G = ox.distance.add_edge_lengths(G)
            data.road_graph = G
            logger.info("Road network: {n} nodes, {e} edges",
                         n=G.number_of_nodes(), e=G.number_of_edges())
        except Exception as exc:
            logger.warning("Could not fetch road network: {e}", e=exc)

        # ── Buildings ───────────────────────────────────────────
        logger.info("Fetching buildings …")
        try:
            buildings = ox.features_from_point(
                center_point, dist=rad,
                tags={"building": True},
            )
            if not buildings.empty:
                buildings = self._extract_building_heights(buildings)
                data.buildings_gdf = buildings
                logger.info("Buildings: {n} features", n=len(buildings))
        except Exception as exc:
            logger.warning("Could not fetch buildings: {e}", e=exc)

        # ── Trees / vegetation ──────────────────────────────────
        logger.info("Fetching trees & vegetation …")
        try:
            trees = ox.features_from_point(
                center_point, dist=rad,
                tags={"natural": ["tree", "tree_row", "wood"],
                      "landuse": ["forest", "grass"]},
            )
            if not trees.empty:
                trees = self._extract_tree_heights(trees)
                data.trees_gdf = trees
                logger.info("Trees/vegetation: {n} features", n=len(trees))
        except Exception as exc:
            logger.warning("Could not fetch trees: {e}", e=exc)

        return data

    def fetch_route_area(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        buffer_m: float = 500.0,
        network_type: str | None = None,
    ) -> UrbanData:
        """
        Fetch data for the bounding area around a route (origin → destination),
        padded by *buffer_m* on each side.
        """
        lat_mid = (origin[0] + destination[0]) / 2
        lon_mid = (origin[1] + destination[1]) / 2
        # Approximate distance between origin and destination
        from shapely.geometry import Point as ShapelyPoint
        dist_approx = ShapelyPoint(origin).distance(ShapelyPoint(destination)) * 111_320  # rough m
        radius = (dist_approx / 2) + buffer_m
        radius = max(radius, 1000.0)

        return self.fetch_area(lat_mid, lon_mid, radius_m=radius, network_type=network_type)

    # ── Height extraction helpers ───────────────────────────────

    @staticmethod
    def _extract_building_heights(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Parse building heights from OSM tags."""
        gdf = gdf.copy()

        def _parse_height(row: Any) -> float:
            # Try explicit height tag first
            if "height" in row.index and pd.notna(row.get("height")):
                try:
                    return float(str(row["height"]).replace("m", "").strip())
                except (ValueError, TypeError):
                    pass
            # Estimate from number of levels
            if "building:levels" in row.index and pd.notna(row.get("building:levels")):
                try:
                    return float(row["building:levels"]) * 3.0  # ~3m per floor
                except (ValueError, TypeError):
                    pass
            return settings.DEFAULT_BUILDING_HEIGHT_M

        import pandas as pd  # local import to avoid name clash
        gdf["estimated_height_m"] = gdf.apply(_parse_height, axis=1)
        return gdf

    @staticmethod
    def _extract_tree_heights(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Parse tree heights from OSM tags or use defaults."""
        gdf = gdf.copy()

        def _parse_height(row: Any) -> float:
            if "height" in row.index and pd.notna(row.get("height")):
                try:
                    return float(str(row["height"]).replace("m", "").strip())
                except (ValueError, TypeError):
                    pass
            return settings.DEFAULT_TREE_HEIGHT_M

        import pandas as pd
        gdf["estimated_height_m"] = gdf.apply(_parse_height, axis=1)
        return gdf
