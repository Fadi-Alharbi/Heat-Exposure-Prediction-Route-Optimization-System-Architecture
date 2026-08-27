"""
Path Finder
───────────
Finds optimal routes through the heat-weighted road network graph.

Supports:
  - Shortest path (Dijkstra/A*) with configurable weight
  - K-shortest paths for route comparison
  - Route statistics aggregation
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


@dataclass
class RouteResult:
    """A single route with its statistics."""
    route_id: str
    path_nodes: list[int]
    total_distance_m: float
    total_time_s: float
    total_time_min: float
    avg_heat_exposure: float
    max_heat_exposure: float
    cumulative_heat_exposure: float
    avg_shade_fraction: float
    combined_cost: float
    label: str = ""            # "fastest" | "coolest" | "balanced" | "Route A" etc.
    is_recommended: bool = False
    segments: list[dict] = field(default_factory=list)


class PathFinder:
    """
    Finds and compares routes in a heat-weighted graph.
    """

    def find_optimal_route(
        self,
        graph: nx.MultiDiGraph,
        origin_node: int,
        destination_node: int,
        weight: str = "combined_cost",
    ) -> RouteResult | None:
        """
        Find the optimal path using Dijkstra's algorithm.

        Parameters
        ----------
        graph : nx.MultiDiGraph
            Weighted road network.
        origin_node : int
            Starting node ID.
        destination_node : int
            Ending node ID.
        weight : str
            Edge attribute to use as weight (e.g., "combined_cost",
            "travel_time_s", "cumulative_heat_exposure").
        """
        try:
            path = nx.shortest_path(
                graph, origin_node, destination_node, weight=weight,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            logger.error("No path found: {e}", e=exc)
            return None

        stats = self._compute_route_stats(graph, path)
        stats.route_id = f"route_{weight}"
        stats.label = weight
        return stats

    def find_k_shortest_paths(
        self,
        graph: nx.MultiDiGraph,
        origin_node: int,
        destination_node: int,
        k: int | None = None,
        weight: str = "combined_cost",
    ) -> list[RouteResult]:
        """
        Find the K shortest (lowest cost) paths.

        Uses Yen's k-shortest simple paths algorithm.
        """
        k = k or settings.K_SHORTEST_PATHS

        try:
            paths = list(nx.shortest_simple_paths(
                graph, origin_node, destination_node, weight=weight,
            ))
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            logger.error("No paths found: {e}", e=exc)
            return []

        results = []
        for i, path in enumerate(paths[:k]):
            stats = self._compute_route_stats(graph, path)
            stats.route_id = f"route_{i+1}"
            stats.label = chr(65 + i)  # A, B, C, ...
            results.append(stats)

        return results

    def compare_routes(
        self,
        graph: nx.MultiDiGraph,
        origin_node: int,
        destination_node: int,
    ) -> list[RouteResult]:
        """
        Find and compare routes using different optimization criteria:
        - Fastest (minimize time)
        - Coolest (minimize heat exposure)
        - Balanced (combined cost)

        Marks the recommended route.
        """
        routes = []

        # Fastest route
        fastest = self.find_optimal_route(graph, origin_node, destination_node, "travel_time_s")
        if fastest:
            fastest.label = "Fastest"
            fastest.route_id = "fastest"
            routes.append(fastest)

        # Coolest route
        coolest = self.find_optimal_route(
            graph, origin_node, destination_node, "cumulative_heat_exposure",
        )
        if coolest:
            coolest.label = "Coolest"
            coolest.route_id = "coolest"
            routes.append(coolest)

        # Balanced route
        balanced = self.find_optimal_route(graph, origin_node, destination_node, "combined_cost")
        if balanced:
            balanced.label = "Balanced"
            balanced.route_id = "balanced"
            routes.append(balanced)

        # Mark recommended (balanced is usually recommended)
        if routes:
            self._mark_recommended(routes)

        return routes

    def find_nearest_node(
        self,
        graph: nx.MultiDiGraph,
        latitude: float,
        longitude: float,
    ) -> int:
        """Find the graph node closest to a given coordinate."""
        import osmnx as ox
        return ox.distance.nearest_nodes(graph, longitude, latitude)

    # ── Route statistics ────────────────────────────────────────

    def _compute_route_stats(
        self,
        graph: nx.MultiDiGraph,
        path: list[int],
    ) -> RouteResult:
        """Aggregate statistics for a path."""
        total_dist = 0.0
        total_time = 0.0
        total_cum_heat = 0.0
        total_combined = 0.0
        heat_scores = []
        shade_fractions = []
        segments = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            # Get the best edge (lowest combined cost)
            edge_data = self._get_best_edge(graph, u, v)
            if edge_data is None:
                continue

            length = edge_data.get("length", 0)
            time_s = edge_data.get("travel_time_s", 0)
            heat = edge_data.get("heat_exposure_score", 0)
            cum_heat = edge_data.get("cumulative_heat_exposure", 0)
            shade = edge_data.get("shade_fraction", 0)
            combined = edge_data.get("combined_cost", 0)

            total_dist += length
            total_time += time_s
            total_cum_heat += cum_heat
            total_combined += combined
            heat_scores.append(heat)
            shade_fractions.append(shade)

            segments.append({
                "from_node": u,
                "to_node": v,
                "length_m": length,
                "time_s": time_s,
                "heat_score": heat,
                "shade_fraction": shade,
                "surface_type": edge_data.get("surface_type", "unknown"),
            })

        avg_heat = np.mean(heat_scores) if heat_scores else 0.0
        max_heat = max(heat_scores) if heat_scores else 0.0
        avg_shade = np.mean(shade_fractions) if shade_fractions else 0.0

        return RouteResult(
            route_id="",
            path_nodes=path,
            total_distance_m=round(total_dist, 1),
            total_time_s=round(total_time, 1),
            total_time_min=round(total_time / 60, 1),
            avg_heat_exposure=round(avg_heat, 2),
            max_heat_exposure=round(max_heat, 2),
            cumulative_heat_exposure=round(total_cum_heat, 2),
            avg_shade_fraction=round(avg_shade, 3),
            combined_cost=round(total_combined, 2),
            segments=segments,
        )

    @staticmethod
    def _get_best_edge(
        graph: nx.MultiDiGraph,
        u: int,
        v: int,
    ) -> dict | None:
        """Get the edge with lowest combined_cost between u and v."""
        if not graph.has_edge(u, v):
            return None
        edges = graph[u][v]
        best_key = min(
            edges.keys(),
            key=lambda k: edges[k].get("combined_cost", float("inf")),
        )
        return edges[best_key]

    @staticmethod
    def _mark_recommended(routes: list[RouteResult]) -> None:
        """
        Mark the recommended route based on a scoring heuristic.

        Preference: significant heat reduction with small time increase.
        """
        if not routes:
            return

        fastest = min(routes, key=lambda r: r.total_time_s)
        coolest = min(routes, key=lambda r: r.cumulative_heat_exposure)

        # If coolest adds < 30% time, recommend it
        if fastest.total_time_s > 0:
            time_ratio = coolest.total_time_s / fastest.total_time_s
            if time_ratio < 1.3:
                coolest.is_recommended = True
                return

        # Otherwise recommend balanced
        balanced = [r for r in routes if r.label == "Balanced"]
        if balanced:
            balanced[0].is_recommended = True
        else:
            routes[0].is_recommended = True
