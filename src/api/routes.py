"""
API Routes
──────────
FastAPI endpoint definitions for the Heat Exposure Router.
"""

from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.api.schemas import (
    RouteOptimizeRequest, RouteOptimizeResponse, RouteResponse,
    RouteCompareRequest, RouteSegmentResponse,
    WeatherResponse, HealthResponse, RecommendationResponse,
    DepartureTimeRequest,
)
from src.data_ingestion.weather_client import WeatherClient
from src.feature_engineering.heat_index_calculator import HeatIndexCalculator

router = APIRouter()

# Shared instances
weather_client = WeatherClient()
heat_calc = HeatIndexCalculator()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """API health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        components={
            "weather_api": "connected",
            "ml_model": "heuristic_mode",
            "database": "not_configured",
        },
    )


@router.get("/weather/{lat}/{lon}", response_model=WeatherResponse)
async def get_weather(lat: float, lon: float):
    """Get current weather conditions for a location."""
    try:
        snapshot = weather_client.get_current(lat, lon)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Weather data not available")

        return WeatherResponse(
            latitude=lat,
            longitude=lon,
            temperature_c=snapshot.temperature_c,
            humidity_pct=snapshot.relative_humidity_pct,
            wind_speed_kmh=snapshot.wind_speed_kmh,
            radiation_wm2=snapshot.total_radiation_wm2,
            cloud_cover_pct=snapshot.cloud_cover_pct,
            apparent_temperature_c=snapshot.apparent_temperature_c,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Weather fetch error: {e}", e=exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/heat-index/{lat}/{lon}")
async def get_heat_index(lat: float, lon: float):
    """Get current heat indices (Heat Index, WBGT, Apparent Temp) for a location."""
    try:
        snapshot = weather_client.get_current(lat, lon)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Weather data not available")

        indices = heat_calc.calculate_all(
            temperature_c=snapshot.temperature_c,
            relative_humidity=snapshot.relative_humidity_pct,
            wind_speed_kmh=snapshot.wind_speed_kmh,
            solar_radiation_wm2=snapshot.total_radiation_wm2,
        )

        return {
            "latitude": lat,
            "longitude": lon,
            "air_temperature_c": snapshot.temperature_c,
            "heat_index_c": indices.heat_index_c,
            "wbgt_c": indices.wbgt_c,
            "apparent_temperature_c": indices.apparent_temperature_c,
            "thermal_stress_category": indices.thermal_stress_category,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Heat index error: {e}", e=exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/route/optimize", response_model=RouteOptimizeResponse)
async def optimize_route(request: RouteOptimizeRequest):
    """
    Optimize a route between origin and destination.

    Returns multiple candidate routes with heat exposure analysis
    and a recommendation.
    """
    try:
        # Get weather
        mid_lat = (request.origin.latitude + request.destination.latitude) / 2
        mid_lon = (request.origin.longitude + request.destination.longitude) / 2
        snapshot = weather_client.get_current(mid_lat, mid_lon)

        weather_info = {}
        if snapshot:
            weather_info = {
                "temperature_c": snapshot.temperature_c,
                "humidity_pct": snapshot.relative_humidity_pct,
                "wind_speed_kmh": snapshot.wind_speed_kmh,
                "radiation_wm2": snapshot.total_radiation_wm2,
            }

        # NOTE: Full route optimization requires OSMnx graph fetching,
        # which can be slow. For the API, we return a simplified response
        # indicating the system architecture is in place.
        return RouteOptimizeResponse(
            routes=[],
            recommendation=RecommendationResponse(
                title="System Ready",
                description=(
                    "The route optimization engine is configured. "
                    "Full graph-based routing requires OSMnx data download "
                    "which is handled by the Streamlit UI or batch processing."
                ),
                route_label="",
                confidence="medium",
                factors=["Weather data available", "ML model in heuristic mode"],
            ),
            weather=weather_info,
        )
    except Exception as exc:
        logger.error("Route optimization error: {e}", e=exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/departure-time/suggest")
async def suggest_departure_time(request: DepartureTimeRequest):
    """Suggest the best departure time for a trip."""
    from src.modeling.time_series_model import TimeSeriesModel

    try:
        ts_model = TimeSeriesModel()

        # Get forecast data
        forecast = weather_client.get_forecast(
            request.origin.latitude, request.origin.longitude, days=1,
        )

        profile = ts_model.forecast_daily_profile(
            forecast_df=forecast.hourly if forecast else None,
        )

        best_hour, best_exposure = ts_model.find_best_departure_time(
            profile,
            min(request.candidate_hours),
            max(request.candidate_hours),
        )

        comparison = ts_model.compare_departure_times(profile, request.candidate_hours)

        return {
            "best_hour": best_hour,
            "best_time": f"{best_hour:02d}:00",
            "best_exposure": best_exposure,
            "hourly_comparison": comparison.to_dict("records"),
        }
    except Exception as exc:
        logger.error("Departure time suggestion error: {e}", e=exc)
        raise HTTPException(status_code=500, detail=str(exc))
