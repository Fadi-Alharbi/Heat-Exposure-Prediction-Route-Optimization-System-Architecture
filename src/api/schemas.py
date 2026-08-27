"""
API Schemas
───────────
Pydantic models for request/response validation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────

class Coordinate(BaseModel):
    """A geographic coordinate."""
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


class RouteOptimizeRequest(BaseModel):
    """Request to optimize a single route."""
    origin: Coordinate
    destination: Coordinate
    departure_time: str | None = Field(None, description="ISO format datetime, e.g. 2026-08-27T14:00:00+03:00")
    user_speed_kmh: float = Field(15.0, gt=0, le=100)
    transport_mode: str = Field("bike", description="walk | bike | scooter | drive")
    preference: str = Field("balanced", description="fastest | coolest | balanced | slightly_cool | slightly_fast")
    alpha: float | None = Field(None, ge=0, le=1, description="Custom time weight")
    beta: float | None = Field(None, ge=0, le=1, description="Custom heat weight")


class RouteCompareRequest(BaseModel):
    """Request to compare routes with different strategies."""
    origin: Coordinate
    destination: Coordinate
    departure_time: str | None = None
    user_speed_kmh: float = 15.0
    transport_mode: str = "bike"


class TripItem(BaseModel):
    """A single trip in a multi-trip request."""
    trip_id: str
    origin: Coordinate
    destination: Coordinate
    earliest_time: str | None = None
    latest_time: str | None = None
    priority: int = 1


class MultiTripRequest(BaseModel):
    """Request to optimize multiple trips."""
    trips: list[TripItem]
    start_time: str | None = None
    transport_mode: str = "bike"


class DepartureTimeRequest(BaseModel):
    """Request for best departure time."""
    origin: Coordinate
    destination: Coordinate
    candidate_hours: list[int] = Field(default_factory=lambda: list(range(6, 21)))


# ── Response Models ─────────────────────────────────────────────

class RouteSegmentResponse(BaseModel):
    """A segment of a route."""
    from_node: int
    to_node: int
    length_m: float
    time_s: float
    heat_score: float
    shade_fraction: float
    surface_type: str


class RouteResponse(BaseModel):
    """A single route result."""
    route_id: str
    label: str
    distance_m: float
    distance_km: float
    time_min: float
    avg_heat_exposure: float
    max_heat_exposure: float
    cumulative_heat_exposure: float
    avg_shade_pct: float
    is_recommended: bool
    segments: list[RouteSegmentResponse] = []


class RecommendationResponse(BaseModel):
    """A recommendation with explanation."""
    title: str
    description: str
    route_label: str
    departure_time: str | None = None
    confidence: str
    factors: list[str] = []


class RouteOptimizeResponse(BaseModel):
    """Response for route optimization."""
    routes: list[RouteResponse]
    recommendation: RecommendationResponse | None = None
    weather: dict = {}
    solar: dict = {}


class WeatherResponse(BaseModel):
    """Current weather data."""
    latitude: float
    longitude: float
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    radiation_wm2: float
    cloud_cover_pct: float
    apparent_temperature_c: float | None = None


class HealthResponse(BaseModel):
    """API health check."""
    status: str
    version: str
    components: dict
