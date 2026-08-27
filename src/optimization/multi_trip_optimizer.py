"""
Multi-Trip Optimizer
────────────────────
Optimizes the order and timing of multiple delivery trips for a worker
who has several destinations to visit throughout the day.

Example:
  A delivery worker has 4 packages to deliver. Instead of following
  the order they were received, the system finds the optimal sequence
  and departure time for each trip to minimize total heat exposure.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np
from loguru import logger

from src.optimization.path_finder import PathFinder, RouteResult
from src.modeling.time_series_model import TimeSeriesModel


@dataclass
class TripRequest:
    """A single delivery/trip request."""
    trip_id: str
    origin_lat: float
    origin_lon: float
    destination_lat: float
    destination_lon: float
    earliest_time: dt.datetime | None = None
    latest_time: dt.datetime | None = None
    priority: int = 1  # higher = more urgent


@dataclass
class ScheduledTrip:
    """A trip with assigned order and departure time."""
    trip_id: str
    order: int
    departure_time: dt.datetime
    route: RouteResult | None
    heat_exposure: float
    estimated_arrival: dt.datetime | None = None


@dataclass
class MultiTripPlan:
    """Complete plan for multiple trips."""
    trips: list[ScheduledTrip]
    total_time_min: float
    total_heat_exposure: float
    total_distance_m: float
    optimization_method: str
    alternative_plans: list[dict] = field(default_factory=list)


class MultiTripOptimizer:
    """
    Optimizes the scheduling of multiple trips for heat-aware delivery.
    """

    def __init__(self):
        self.path_finder = PathFinder()
        self.time_model = TimeSeriesModel()

    def optimize_trips(
        self,
        trips: list[TripRequest],
        graph: Any,  # nx.MultiDiGraph
        start_time: dt.datetime,
        base_temp_c: float = 40.0,
        max_permutations: int = 120,
    ) -> MultiTripPlan:
        """
        Find the optimal order and timing for a set of trips.

        Strategy:
        1. Generate candidate orderings (up to max_permutations)
        2. For each ordering, simulate timing and heat exposure
        3. Select the ordering with lowest total heat exposure

        Parameters
        ----------
        trips : list of TripRequest
            The trips to schedule.
        graph : MultiDiGraph
            Heat-weighted road graph.
        start_time : datetime
            Earliest possible start time.
        base_temp_c : float
            Base daily temperature for heat forecasting.
        max_permutations : int
            Maximum orderings to evaluate (capped for performance).
        """
        n_trips = len(trips)
        if n_trips == 0:
            return MultiTripPlan([], 0, 0, 0, "none")

        if n_trips == 1:
            return self._single_trip_plan(trips[0], graph, start_time)

        # Generate hourly heat profile for the day
        hourly_profile = self.time_model.forecast_daily_profile(
            base_temp_c=base_temp_c,
        )
        heat_by_hour = {c.hour: c.heat_exposure_estimate for c in hourly_profile}

        # Generate candidate orderings
        if n_trips <= 5:
            orderings = list(permutations(range(n_trips)))
        else:
            # For many trips, use greedy + random sampling
            orderings = self._sample_orderings(n_trips, max_permutations)

        logger.info(
            "Evaluating {n} orderings for {t} trips",
            n=len(orderings), t=n_trips,
        )

        best_plan = None
        best_score = float("inf")
        alternative_scores = []

        for ordering in orderings:
            plan = self._evaluate_ordering(
                trips, ordering, graph, start_time, heat_by_hour,
            )
            score = plan.total_heat_exposure

            alternative_scores.append({
                "ordering": [trips[i].trip_id for i in ordering],
                "total_heat": plan.total_heat_exposure,
                "total_time_min": plan.total_time_min,
            })

            if score < best_score:
                best_score = score
                best_plan = plan

        if best_plan:
            best_plan.optimization_method = f"exhaustive ({len(orderings)} orderings)"
            # Top 3 alternatives
            alternative_scores.sort(key=lambda x: x["total_heat"])
            best_plan.alternative_plans = alternative_scores[:3]

        return best_plan or MultiTripPlan([], 0, 0, 0, "failed")

    def suggest_departure_time(
        self,
        trip: TripRequest,
        graph: Any,
        candidate_hours: list[int] | None = None,
        base_temp_c: float = 40.0,
    ) -> dict:
        """
        Suggest the best departure time for a single trip.

        Returns dict with best_hour, exposure at each candidate hour.
        """
        hours = candidate_hours or list(range(6, 21))
        profile = self.time_model.forecast_daily_profile(base_temp_c=base_temp_c)

        best_hour, best_exposure = self.time_model.find_best_departure_time(
            profile, min(hours), max(hours),
        )

        comparison = self.time_model.compare_departure_times(profile, hours)

        return {
            "best_hour": best_hour,
            "best_exposure": best_exposure,
            "comparison": comparison.to_dict("records"),
        }

    # ── Internal methods ────────────────────────────────────────

    def _single_trip_plan(
        self,
        trip: TripRequest,
        graph: Any,
        start_time: dt.datetime,
    ) -> MultiTripPlan:
        """Create a plan for a single trip."""
        origin = self.path_finder.find_nearest_node(
            graph, trip.origin_lat, trip.origin_lon,
        )
        dest = self.path_finder.find_nearest_node(
            graph, trip.destination_lat, trip.destination_lon,
        )
        route = self.path_finder.find_optimal_route(graph, origin, dest)

        scheduled = ScheduledTrip(
            trip_id=trip.trip_id,
            order=1,
            departure_time=start_time,
            route=route,
            heat_exposure=route.cumulative_heat_exposure if route else 0,
        )

        return MultiTripPlan(
            trips=[scheduled],
            total_time_min=route.total_time_min if route else 0,
            total_heat_exposure=route.cumulative_heat_exposure if route else 0,
            total_distance_m=route.total_distance_m if route else 0,
            optimization_method="single_trip",
        )

    def _evaluate_ordering(
        self,
        trips: list[TripRequest],
        ordering: tuple[int, ...],
        graph: Any,
        start_time: dt.datetime,
        heat_by_hour: dict[int, float],
    ) -> MultiTripPlan:
        """Evaluate a specific trip ordering."""
        scheduled_trips = []
        current_time = start_time
        total_time = 0.0
        total_heat = 0.0
        total_dist = 0.0

        for order_idx, trip_idx in enumerate(ordering):
            trip = trips[trip_idx]

            # Get heat level at current hour
            hour = current_time.hour
            hour_heat = heat_by_hour.get(hour, 30.0)

            # Estimate trip duration and heat (simplified)
            # In production, would run full path finding for each trip
            est_duration_min = 15.0  # placeholder
            est_heat = hour_heat * est_duration_min / 60.0

            arrival = current_time + dt.timedelta(minutes=est_duration_min)

            scheduled_trips.append(ScheduledTrip(
                trip_id=trip.trip_id,
                order=order_idx + 1,
                departure_time=current_time,
                route=None,
                heat_exposure=est_heat,
                estimated_arrival=arrival,
            ))

            total_time += est_duration_min
            total_heat += est_heat
            current_time = arrival + dt.timedelta(minutes=5)  # 5min between trips

        return MultiTripPlan(
            trips=scheduled_trips,
            total_time_min=total_time,
            total_heat_exposure=total_heat,
            total_distance_m=total_dist,
            optimization_method="evaluation",
        )

    @staticmethod
    def _sample_orderings(n: int, max_samples: int) -> list[tuple[int, ...]]:
        """Generate sampled orderings for large trip counts."""
        rng = np.random.RandomState(42)
        orderings = set()

        # Always include natural order
        orderings.add(tuple(range(n)))

        # Random permutations
        for _ in range(max_samples * 2):
            perm = list(range(n))
            rng.shuffle(perm)
            orderings.add(tuple(perm))
            if len(orderings) >= max_samples:
                break

        return list(orderings)
