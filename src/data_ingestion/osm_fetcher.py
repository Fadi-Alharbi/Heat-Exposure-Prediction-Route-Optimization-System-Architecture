"""
OpenStreetMap Data Fetcher
──────────────────────────
Uses OSMnx to download road networks, buildings, and tree data
for a given location, then converts them to analysis-ready structures.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from loguru import logger
from shapely.geometry import Point
from shapely.ops import unary_union

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

    def fetch_from_geopackage(self, gpkg_path: str) -> UrbanData:
        """
        Read road network, buildings, and trees from a local GeoPackage file (e.g. from BBBike).
        This provides offline, lightning-fast data loading.
        """
        logger.info("Loading map data locally from GeoPackage: {path}", path=gpkg_path)
        data = UrbanData()
        
        try:
            import fiona
            layers = fiona.listlayers(gpkg_path)
        except Exception as e:
            logger.error("Could not read GeoPackage layers: {e}", e=e)
            return data

        # ── Road network ────────────────────────────────────────
        if 'lines' in layers:
            logger.info("Extracting road network from GeoPackage...")
            gdf_lines = gpd.read_file(gpkg_path, layer='lines')
            
            if 'highway' in gdf_lines.columns:
                G = nx.MultiDiGraph()
                roads = gdf_lines.dropna(subset=['highway', 'geometry']).copy()

                # BBBike exports ways as long lines.  Their intersections are
                # often vertices in the middle of a line, not its endpoints;
                # using endpoints directly produces thousands of tiny,
                # disconnected graphs.  Node the local lines at intersections
                # before creating the routing graph.
                noded_roads = unary_union(list(roads.geometry))
                segments = list(noded_roads.geoms) if hasattr(noded_roads, "geoms") else [noded_roads]

                for geom in segments:
                    if geom is None or geom.is_empty or geom.geom_type != 'LineString':
                        continue

                    # Keep the attributes of an intersecting source way.
                    try:
                        matches = roads.iloc[roads.sindex.query(geom, predicate="intersects")]
                        row = matches.iloc[0] if not matches.empty else None
                    except Exception:
                        row = None

                    coords = list(geom.coords)
                    start_pt, end_pt = coords[0], coords[-1]
                    
                    u = f"{start_pt[0]:.6f}_{start_pt[1]:.6f}"
                    v = f"{end_pt[0]:.6f}_{end_pt[1]:.6f}"
                    
                    if u not in G: G.add_node(u, x=start_pt[0], y=start_pt[1])
                    if v not in G: G.add_node(v, x=end_pt[0], y=end_pt[1])
                    
                    # Calculate approximate length in meters using Haversine formula
                    _lon1, _lat1, _lon2, _lat2 = math.radians(start_pt[0]), math.radians(start_pt[1]), math.radians(end_pt[0]), math.radians(end_pt[1])
                    _dlon = _lon2 - _lon1
                    _dlat = _lat2 - _lat1
                    _a = math.sin(_dlat/2)**2 + math.cos(_lat1) * math.cos(_lat2) * math.sin(_dlon/2)**2
                    length = 2 * math.asin(math.sqrt(_a)) * 6371000  # Earth radius in meters
                    
                    edge_attrs = {
                        "length": length,
                        "highway": row["highway"] if row is not None else "unknown",
                        "name": row.get("name", "") if row is not None else "",
                        "geometry": geom
                    }
                    
                    G.add_edge(u, v, **edge_attrs)
                    # This application routes people walking/cycling; make
                    # local street segments traversable both ways so the
                    # GeoPackage supplies a usable pedestrian network.
                    G.add_edge(v, u, **edge_attrs)
                
                data.road_graph = G
                logger.info("Road network loaded: {n} nodes, {e} edges", n=G.number_of_nodes(), e=G.number_of_edges())
                
                # set center of map based on first node
                if len(G.nodes) > 0:
                    first_node = list(G.nodes(data=True))[0][1]
                    data.center = (first_node['y'], first_node['x'])

        # ── Buildings ───────────────────────────────────────────
        if 'multipolygons' in layers:
            logger.info("Extracting buildings and trees from GeoPackage...")
            gdf_poly = gpd.read_file(gpkg_path, layer='multipolygons')
            
            if 'building' in gdf_poly.columns:
                buildings = gdf_poly[gdf_poly['building'].notna()].copy()
                if not buildings.empty:
                    buildings = self._extract_building_heights(buildings)
                    data.buildings_gdf = buildings
                    logger.info("Buildings loaded: {n} features", n=len(buildings))

            if 'natural' in gdf_poly.columns:
                trees = gdf_poly[gdf_poly['natural'].isin(["tree", "wood", "tree_row"])].copy()
                if not trees.empty:
                    trees = self._extract_tree_heights(trees)
                    data.trees_gdf = trees
                    logger.info("Trees loaded: {n} features", n=len(trees))

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
