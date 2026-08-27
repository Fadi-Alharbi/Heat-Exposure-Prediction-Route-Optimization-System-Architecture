"""Tests for the Path Finder."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import pytest
from src.optimization.path_finder import PathFinder, RouteResult


class TestPathFinder:
    """Test path finding algorithms."""

    def setup_method(self):
        """Create a simple test graph."""
        self.G = nx.MultiDiGraph()

        # Path 1: Fast but hot (0 → 1 → 3)
        self.G.add_edge(0, 1, length=500, travel_time_s=60, heat_exposure_score=40,
                        cumulative_heat_exposure=40, shade_fraction=0.1,
                        combined_cost=28, surface_type="asphalt")
        self.G.add_edge(1, 3, length=500, travel_time_s=60, heat_exposure_score=45,
                        cumulative_heat_exposure=45, shade_fraction=0.05,
                        combined_cost=31, surface_type="asphalt")

        # Path 2: Slower but cooler (0 → 2 → 3)
        self.G.add_edge(0, 2, length=600, travel_time_s=80, heat_exposure_score=20,
                        cumulative_heat_exposure=20, shade_fraction=0.6,
                        combined_cost=18, surface_type="paving_stones")
        self.G.add_edge(2, 3, length=600, travel_time_s=80, heat_exposure_score=15,
                        cumulative_heat_exposure=15, shade_fraction=0.7,
                        combined_cost=13, surface_type="grass")

        # Add node positions
        self.G.nodes[0]["y"], self.G.nodes[0]["x"] = 24.7136, 46.6753
        self.G.nodes[1]["y"], self.G.nodes[1]["x"] = 24.7150, 46.6780
        self.G.nodes[2]["y"], self.G.nodes[2]["x"] = 24.7120, 46.6790
        self.G.nodes[3]["y"], self.G.nodes[3]["x"] = 24.7100, 46.6830

        self.finder = PathFinder()

    def test_find_fastest_route(self):
        route = self.finder.find_optimal_route(self.G, 0, 3, "travel_time_s")
        assert route is not None
        assert route.path_nodes == [0, 1, 3]  # faster path
        assert route.total_time_s == 120

    def test_find_coolest_route(self):
        route = self.finder.find_optimal_route(self.G, 0, 3, "cumulative_heat_exposure")
        assert route is not None
        assert route.path_nodes == [0, 2, 3]  # cooler path
        assert route.cumulative_heat_exposure == 35

    def test_find_balanced_route(self):
        route = self.finder.find_optimal_route(self.G, 0, 3, "combined_cost")
        assert route is not None
        assert route.path_nodes == [0, 2, 3]  # balanced favors cooler

    def test_compare_routes(self):
        routes = self.finder.compare_routes(self.G, 0, 3)
        assert len(routes) >= 2  # at least fastest and coolest
        labels = [r.label for r in routes]
        assert "Fastest" in labels
        assert "Coolest" in labels

    def test_recommended_route_is_marked(self):
        routes = self.finder.compare_routes(self.G, 0, 3)
        recommended = [r for r in routes if r.is_recommended]
        assert len(recommended) == 1

    def test_no_path_returns_none(self):
        G = nx.MultiDiGraph()
        G.add_node(0)
        G.add_node(1)
        # No edges
        route = self.finder.find_optimal_route(G, 0, 1)
        assert route is None

    def test_route_stats_computed(self):
        route = self.finder.find_optimal_route(self.G, 0, 3, "combined_cost")
        assert route.total_distance_m > 0
        assert route.total_time_min > 0
        assert route.avg_heat_exposure >= 0
        assert 0 <= route.avg_shade_fraction <= 1
