"""
Weight Assigner
───────────────
Assigns and manages edge weights for the optimization graph.

The combined cost function:
    cost = α × travel_time + β × heat_exposure

Where:
  - α controls importance of speed
  - β controls importance of thermal comfort
  - The user can adjust the balance via a preference slider
"""

from __future__ import annotations

import networkx as nx
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


class WeightAssigner:
    """
    Manages edge weight assignment for multi-objective route optimization.
    """

    def __init__(self, alpha: float | None = None, beta: float | None = None):
        self.alpha = alpha if alpha is not None else settings.ALPHA_TIME
        self.beta = beta if beta is not None else settings.BETA_HEAT
        self._normalize()

    def _normalize(self) -> None:
        """Ensure α + β = 1."""
        total = self.alpha + self.beta
        if total > 0:
            self.alpha /= total
            self.beta /= total

    def set_preference(self, preference: str) -> None:
        """
        Set weights from a named preference.

        Options: "fastest", "coolest", "balanced",
                 "slightly_cool", "slightly_fast"
        """
        presets = {
            "fastest": (1.0, 0.0),
            "coolest": (0.0, 1.0),
            "balanced": (0.5, 0.5),
            "slightly_cool": (0.3, 0.7),
            "slightly_fast": (0.7, 0.3),
        }
        if preference in presets:
            self.alpha, self.beta = presets[preference]
            logger.info("Preference set to '{p}': α={a}, β={b}",
                        p=preference, a=self.alpha, b=self.beta)
        else:
            logger.warning("Unknown preference '{p}', using balanced", p=preference)
            self.alpha, self.beta = 0.5, 0.5

    def compute_combined_cost(
        self,
        travel_time_s: float,
        heat_exposure_score: float,
        segment_duration_min: float,
    ) -> float:
        """
        Compute combined cost for a single edge/segment.

        The heat exposure is cumulative: score × duration.
        """
        cumulative_heat = heat_exposure_score * segment_duration_min
        return self.alpha * (travel_time_s / 60.0) + self.beta * (cumulative_heat / 50.0)

    def update_graph_weights(
        self,
        graph: nx.MultiDiGraph,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> nx.MultiDiGraph:
        """
        Recalculate combined_cost on all edges with new weights.

        Assumes edges already have 'travel_time_s' and 'cumulative_heat_exposure'.
        """
        a = alpha if alpha is not None else self.alpha
        b = beta if beta is not None else self.beta
        total = a + b
        if total > 0:
            a, b = a / total, b / total

        for u, v, key, data in graph.edges(keys=True, data=True):
            time_s = data.get("travel_time_s", 0)
            cum_heat = data.get("cumulative_heat_exposure", 0)
            data["combined_cost"] = a * (time_s / 60.0) + b * (cum_heat / 50.0)

        logger.info("Updated graph weights: α={a:.2f}, β={b:.2f}", a=a, b=b)
        return graph

    @staticmethod
    def get_weight_for_mode(mode: str) -> str:
        """
        Return the edge weight attribute name for different optimization modes.

        Returns the NetworkX edge attribute to use as weight in pathfinding.
        """
        modes = {
            "fastest": "travel_time_s",
            "coolest": "cumulative_heat_exposure",
            "balanced": "combined_cost",
            "shortest": "length",
        }
        return modes.get(mode, "combined_cost")
