"""Tests for the Graph Builder."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import pytest

from src.optimization.weight_assigner import WeightAssigner


class TestWeightAssigner:
    """Test weight assignment and preference presets."""

    def test_default_weights(self):
        wa = WeightAssigner()
        assert 0 <= wa.alpha <= 1
        assert 0 <= wa.beta <= 1
        assert abs(wa.alpha + wa.beta - 1.0) < 0.01

    def test_fastest_preset(self):
        wa = WeightAssigner()
        wa.set_preference("fastest")
        assert wa.alpha == 1.0
        assert wa.beta == 0.0

    def test_coolest_preset(self):
        wa = WeightAssigner()
        wa.set_preference("coolest")
        assert wa.alpha == 0.0
        assert wa.beta == 1.0

    def test_balanced_preset(self):
        wa = WeightAssigner()
        wa.set_preference("balanced")
        assert wa.alpha == 0.5
        assert wa.beta == 0.5

    def test_combined_cost_calculation(self):
        wa = WeightAssigner(alpha=0.5, beta=0.5)
        cost = wa.compute_combined_cost(
            travel_time_s=60.0,
            heat_exposure_score=30.0,
            segment_duration_min=1.0,
        )
        # 0.5 * 1 minute + 0.5 * (30 / 50 heat-equivalent minutes) = 0.8
        assert abs(cost - 0.8) < 0.1


class TestGraphWeighting:
    """Test graph weight updates."""

    def test_update_graph_weights(self):
        G = nx.MultiDiGraph()
        G.add_edge(0, 1, travel_time_s=60, cumulative_heat_exposure=30)
        G.add_edge(1, 2, travel_time_s=120, cumulative_heat_exposure=10)

        wa = WeightAssigner(alpha=0.5, beta=0.5)
        G = wa.update_graph_weights(G)

        e01 = G[0][1][0]
        assert "combined_cost" in e01
        assert e01["combined_cost"] > 0

    def test_weight_mode_lookup(self):
        assert WeightAssigner.get_weight_for_mode("fastest") == "travel_time_s"
        assert WeightAssigner.get_weight_for_mode("coolest") == "cumulative_heat_exposure"
        assert WeightAssigner.get_weight_for_mode("balanced") == "combined_cost"
