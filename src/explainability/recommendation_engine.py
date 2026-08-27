"""
Recommendation Engine
─────────────────────
Generates natural-language recommendations for route and time selection,
explaining WHY a particular option is better.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from src.optimization.path_finder import RouteResult


@dataclass
class Recommendation:
    """A structured recommendation with explanation."""
    title: str
    description: str
    route_label: str
    departure_time: str | None = None
    confidence: str = "high"       # "high" | "medium" | "low"
    factors: list[str] | None = None


class RecommendationEngine:
    """
    Generates actionable, explainable recommendations for the user.
    """

    def recommend_route(
        self,
        routes: list[RouteResult],
        trip_time: dt.datetime | None = None,
    ) -> Recommendation:
        """
        Generate a route recommendation from compared routes.
        """
        if not routes:
            return Recommendation(
                title="No routes available",
                description="Could not find any valid routes for this trip.",
                route_label="N/A",
                confidence="low",
            )

        recommended = next((r for r in routes if r.is_recommended), routes[0])
        fastest = min(routes, key=lambda r: r.total_time_s)
        coolest = min(routes, key=lambda r: r.cumulative_heat_exposure)

        factors = self._identify_factors(recommended, routes)

        # Build description
        if recommended == fastest:
            desc = (
                f"Route {recommended.label} is the fastest option at "
                f"{recommended.total_time_min:.0f} minutes and also has "
                f"reasonable heat exposure."
            )
        else:
            time_diff = recommended.total_time_min - fastest.total_time_min
            if fastest.cumulative_heat_exposure > 0:
                heat_pct = abs(
                    (fastest.cumulative_heat_exposure - recommended.cumulative_heat_exposure)
                    / fastest.cumulative_heat_exposure * 100
                )
            else:
                heat_pct = 0

            desc = (
                f"Route {recommended.label} takes {time_diff:.0f} extra minutes "
                f"({recommended.total_time_min:.0f} min total) but reduces heat "
                f"exposure by {heat_pct:.0f}%. "
            )
            if recommended.avg_shade_fraction > 0.3:
                desc += (
                    f"This route has {recommended.avg_shade_fraction*100:.0f}% average shade coverage."
                )

        time_str = trip_time.strftime("%H:%M") if trip_time else None

        return Recommendation(
            title=f"Route {recommended.label} Recommended",
            description=desc,
            route_label=recommended.label,
            departure_time=time_str,
            confidence="high" if len(routes) >= 3 else "medium",
            factors=factors,
        )

    def recommend_departure_time(
        self,
        best_hour: int,
        current_hour: int,
        best_exposure: float,
        current_exposure: float,
    ) -> Recommendation:
        """
        Generate a departure time recommendation.
        """
        if best_hour == current_hour:
            return Recommendation(
                title="Current time is optimal",
                description=(
                    f"The current time ({current_hour:02d}:00) is already the best "
                    f"option for minimal heat exposure."
                ),
                route_label="",
                departure_time=f"{current_hour:02d}:00",
                confidence="high",
            )

        if current_exposure > 0:
            reduction_pct = abs(
                (current_exposure - best_exposure) / current_exposure * 100
            )
        else:
            reduction_pct = 0

        if best_hour > current_hour:
            wait_hours = best_hour - current_hour
            desc = (
                f"Waiting {wait_hours} hour(s) until {best_hour:02d}:00 would reduce "
                f"heat exposure by approximately {reduction_pct:.0f}%. "
                f"Expected exposure drops from {current_exposure:.1f} to {best_exposure:.1f}."
            )
        else:
            desc = (
                f"The optimal time was {best_hour:02d}:00 (earlier today). "
                f"Current conditions result in higher exposure ({current_exposure:.1f} vs "
                f"{best_exposure:.1f} at optimal time)."
            )

        return Recommendation(
            title=f"Recommended Departure: {best_hour:02d}:00",
            description=desc,
            route_label="",
            departure_time=f"{best_hour:02d}:00",
            confidence="high" if reduction_pct > 20 else "medium",
            factors=[
                f"Heat exposure at {best_hour:02d}:00: {best_exposure:.1f}",
                f"Heat exposure at {current_hour:02d}:00: {current_exposure:.1f}",
                f"Reduction: {reduction_pct:.0f}%",
            ],
        )

    @staticmethod
    def _identify_factors(
        recommended: RouteResult,
        all_routes: list[RouteResult],
    ) -> list[str]:
        """Identify the key factors making this route better."""
        factors = []

        others_avg_heat = sum(r.avg_heat_exposure for r in all_routes) / len(all_routes)
        if recommended.avg_heat_exposure < others_avg_heat * 0.8:
            factors.append("Significantly lower heat exposure than alternatives")

        if recommended.avg_shade_fraction > 0.4:
            factors.append(f"Good shade coverage ({recommended.avg_shade_fraction*100:.0f}%)")
        elif recommended.avg_shade_fraction > 0.2:
            factors.append(f"Moderate shade coverage ({recommended.avg_shade_fraction*100:.0f}%)")

        fastest = min(all_routes, key=lambda r: r.total_time_s)
        if recommended.total_time_s <= fastest.total_time_s * 1.1:
            factors.append("Minimal time overhead (< 10% longer)")
        elif recommended.total_time_s <= fastest.total_time_s * 1.2:
            factors.append("Acceptable time overhead (< 20% longer)")

        return factors
