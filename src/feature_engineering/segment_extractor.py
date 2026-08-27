"""
Segment Extractor
─────────────────
Divides road network edges into smaller segments (~50-100m each) and
enriches them with positional, directional, and temporal attributes
needed for per-segment heat exposure prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import networkx as nx
from loguru import logger
from shapely.geometry import LineString, Point

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


@dataclass
class RoadSegment:
    """A small piece of a road edge with enriched attributes."""
    segment_id: int
    edge_id: tuple               # (u, v, key) from NetworkX
    center_lat: float
    center_lon: float
    length_m: float
    bearing_deg: float           # compass direction of the segment
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    surface_type: str = "unknown"
    highway_type: str = "unknown"
    width_m: float = 6.0        # estimated road width


class SegmentExtractor:
    """
    Splits road network edges into fixed-length segments for
    fine-grained heat exposure analysis.
    """

    def __init__(self, segment_length_m: float | None = None):
        self.segment_length = segment_length_m or settings.SEGMENT_LENGTH_METERS

    def extract_segments(
        self,
        graph: nx.MultiDiGraph,
        user_speed_kmh: float | None = None,
    ) -> list[RoadSegment]:
        """
        Extract segments from all edges in the road network graph.

        Parameters
        ----------
        graph : nx.MultiDiGraph
            OSMnx road network graph with edge geometry.
        user_speed_kmh : float, optional
            User's travel speed (for time estimation).

        Returns
        -------
        list of RoadSegment
        """
        speed = user_speed_kmh or settings.DEFAULT_USER_SPEED_KMH
        segments: list[RoadSegment] = []
        seg_id = 0

        for u, v, key, data in graph.edges(keys=True, data=True):
            edge_length = data.get("length", 0.0)  # meters
            if edge_length < 1.0:
                continue

            # Get geometry (LineString) or create one from node coords
            geometry = data.get("geometry")
            if geometry is None:
                u_data = graph.nodes[u]
                v_data = graph.nodes[v]
                geometry = LineString([
                    (u_data["x"], u_data["y"]),
                    (v_data["x"], v_data["y"]),
                ])

            # Get surface and highway tags
            surface = data.get("surface", "unknown")
            highway = data.get("highway", "unknown")
            if isinstance(surface, list):
                surface = surface[0]
            if isinstance(highway, list):
                highway = highway[0]

            # Estimate road width
            width = self._estimate_width(highway, data)

            # Split edge into segments
            n_segments = max(1, int(math.ceil(edge_length / self.segment_length)))
            actual_segment_length = edge_length / n_segments

            for i in range(n_segments):
                frac_start = i / n_segments
                frac_end = (i + 1) / n_segments
                frac_mid = (frac_start + frac_end) / 2

                # Interpolate positions along the geometry
                try:
                    pt_start = geometry.interpolate(frac_start, normalized=True)
                    pt_end = geometry.interpolate(frac_end, normalized=True)
                    pt_mid = geometry.interpolate(frac_mid, normalized=True)
                except Exception:
                    continue

                # Calculate bearing
                bearing = self._bearing(pt_start.y, pt_start.x, pt_end.y, pt_end.x)

                segments.append(RoadSegment(
                    segment_id=seg_id,
                    edge_id=(u, v, key),
                    center_lat=pt_mid.y,
                    center_lon=pt_mid.x,
                    length_m=actual_segment_length,
                    bearing_deg=bearing,
                    start_lat=pt_start.y,
                    start_lon=pt_start.x,
                    end_lat=pt_end.y,
                    end_lon=pt_end.x,
                    surface_type=str(surface),
                    highway_type=str(highway),
                    width_m=width,
                ))
                seg_id += 1

        logger.info(
            "Extracted {n} segments from {e} edges (target length: {l}m)",
            n=len(segments), e=graph.number_of_edges(), l=self.segment_length,
        )
        return segments

    def extract_path_segments(
        self,
        graph: nx.MultiDiGraph,
        path_nodes: list[int],
    ) -> list[RoadSegment]:
        """Extract segments only for edges along a specific path."""
        subgraph_edges = set()
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            subgraph_edges.add((u, v))

        segments = []
        seg_id = 0

        for u, v, key, data in graph.edges(keys=True, data=True):
            if (u, v) not in subgraph_edges and (v, u) not in subgraph_edges:
                continue

            edge_length = data.get("length", 0.0)
            if edge_length < 1.0:
                continue

            geometry = data.get("geometry")
            if geometry is None:
                u_data = graph.nodes[u]
                v_data = graph.nodes[v]
                geometry = LineString([
                    (u_data["x"], u_data["y"]),
                    (v_data["x"], v_data["y"]),
                ])

            surface = data.get("surface", "unknown")
            highway = data.get("highway", "unknown")
            if isinstance(surface, list):
                surface = surface[0]
            if isinstance(highway, list):
                highway = highway[0]

            width = self._estimate_width(highway, data)

            n_segments = max(1, int(math.ceil(edge_length / self.segment_length)))
            actual_length = edge_length / n_segments

            for i in range(n_segments):
                frac_start = i / n_segments
                frac_end = (i + 1) / n_segments
                frac_mid = (frac_start + frac_end) / 2

                try:
                    pt_start = geometry.interpolate(frac_start, normalized=True)
                    pt_end = geometry.interpolate(frac_end, normalized=True)
                    pt_mid = geometry.interpolate(frac_mid, normalized=True)
                except Exception:
                    continue

                bearing = self._bearing(pt_start.y, pt_start.x, pt_end.y, pt_end.x)

                segments.append(RoadSegment(
                    segment_id=seg_id,
                    edge_id=(u, v, key),
                    center_lat=pt_mid.y,
                    center_lon=pt_mid.x,
                    length_m=actual_length,
                    bearing_deg=bearing,
                    start_lat=pt_start.y,
                    start_lon=pt_start.x,
                    end_lat=pt_end.y,
                    end_lon=pt_end.x,
                    surface_type=str(surface),
                    highway_type=str(highway),
                    width_m=width,
                ))
                seg_id += 1

        return segments

    def segments_to_dataframe(self, segments: list[RoadSegment]) -> pd.DataFrame:
        """Convert a list of RoadSegments to a DataFrame."""
        return pd.DataFrame([vars(s) for s in segments])

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate compass bearing from point 1 to point 2."""
        dlon = math.radians(lon2 - lon1)
        lat1r, lat2r = math.radians(lat1), math.radians(lat2)
        x = math.sin(dlon) * math.cos(lat2r)
        y = (math.cos(lat1r) * math.sin(lat2r)
             - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    @staticmethod
    def _estimate_width(highway_type: str, edge_data: dict) -> float:
        """Estimate road width from highway type."""
        WIDTH_MAP = {
            "motorway": 12.0,
            "trunk": 10.0,
            "primary": 8.0,
            "secondary": 7.0,
            "tertiary": 6.0,
            "residential": 6.0,
            "service": 4.0,
            "footway": 2.0,
            "cycleway": 2.5,
            "path": 2.0,
            "pedestrian": 4.0,
            "living_street": 5.0,
        }
        # Check explicit width tag first
        if "width" in edge_data:
            try:
                return float(str(edge_data["width"]).replace("m", "").strip())
            except (ValueError, TypeError):
                pass
        return WIDTH_MAP.get(str(highway_type).lower(), 6.0)
