"""
Route Comparator
────────────────
Generates side-by-side comparisons of multiple candidate routes,
highlighting trade-offs between speed and heat exposure.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.optimization.path_finder import RouteResult


class RouteComparator:
    """
    Compares multiple routes and generates comparison tables and insights.
    """

    def compare(self, routes: list[RouteResult]) -> pd.DataFrame:
        """
        Create a comparison DataFrame for multiple routes.

        Returns a DataFrame with one row per route and columns for
        all key metrics.
        """
        if not routes:
            return pd.DataFrame()

        rows = []
        for route in routes:
            rows.append({
                "route": route.label or route.route_id,
                "distance_m": route.total_distance_m,
                "distance_km": round(route.total_distance_m / 1000, 2),
                "time_min": route.total_time_min,
                "avg_heat_exposure": route.avg_heat_exposure,
                "max_heat_exposure": route.max_heat_exposure,
                "cumulative_heat": route.cumulative_heat_exposure,
                "avg_shade": round(route.avg_shade_fraction * 100, 1),
                "recommended": "✓" if route.is_recommended else "",
            })

        df = pd.DataFrame(rows)
        return df

    def compute_trade_offs(self, routes: list[RouteResult]) -> list[dict]:
        """
        Compute pairwise trade-offs between routes.

        For each pair, shows how much extra time buys how much
        heat reduction (or vice versa).
        """
        if len(routes) < 2:
            return []

        # Sort by time
        sorted_routes = sorted(routes, key=lambda r: r.total_time_s)
        fastest = sorted_routes[0]

        trade_offs = []
        for route in sorted_routes[1:]:
            time_diff_min = route.total_time_min - fastest.total_time_min
            heat_diff = fastest.cumulative_heat_exposure - route.cumulative_heat_exposure

            if fastest.cumulative_heat_exposure > 0:
                heat_reduction_pct = (heat_diff / fastest.cumulative_heat_exposure) * 100
            else:
                heat_reduction_pct = 0

            trade_offs.append({
                "route": route.label,
                "vs_fastest": fastest.label,
                "extra_time_min": round(time_diff_min, 1),
                "heat_reduction": round(heat_diff, 2),
                "heat_reduction_pct": round(heat_reduction_pct, 1),
                "worth_it": heat_reduction_pct > 15 and time_diff_min < 10,
            })

        return trade_offs

    def generate_summary(self, routes: list[RouteResult]) -> str:
        """
        Generate a human-readable summary of the route comparison.
        """
        if not routes:
            return "No routes to compare."

        recommended = next((r for r in routes if r.is_recommended), routes[0])
        fastest = min(routes, key=lambda r: r.total_time_s)
        coolest = min(routes, key=lambda r: r.cumulative_heat_exposure)

        lines = []
        lines.append("═══ Route Comparison Summary ═══\n")

        for route in routes:
            marker = " ⭐ RECOMMENDED" if route.is_recommended else ""
            lines.append(
                f"  {route.label}: {route.total_time_min} min | "
                f"Heat: {route.avg_heat_exposure:.1f} | "
                f"Shade: {route.avg_shade_fraction*100:.0f}%{marker}"
            )

        lines.append("")

        # Recommendation explanation
        if recommended != fastest:
            time_diff = recommended.total_time_min - fastest.total_time_min
            if fastest.cumulative_heat_exposure > 0:
                heat_pct = (
                    (fastest.cumulative_heat_exposure - recommended.cumulative_heat_exposure)
                    / fastest.cumulative_heat_exposure * 100
                )
            else:
                heat_pct = 0

            lines.append(
                f"  ➤ {recommended.label} is recommended.\n"
                f"    It takes {time_diff:.0f} extra minutes but reduces\n"
                f"    heat exposure by {heat_pct:.0f}%."
            )
        else:
            lines.append(f"  ➤ {recommended.label} is recommended (fastest & coolest).")

        return "\n".join(lines)
