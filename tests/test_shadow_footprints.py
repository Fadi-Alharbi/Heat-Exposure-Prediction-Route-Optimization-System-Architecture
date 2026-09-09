"""Tests for footprint-based building shadow geometry."""

import datetime as dt

from shapely.geometry import LineString, Polygon

from src.feature_engineering.shadow_estimator import ShadowEstimator
from src.feature_engineering.solar_calculator import SolarPosition


def _sun(azimuth_deg: float) -> SolarPosition:
    return SolarPosition(
        timestamp=dt.datetime(2026, 9, 9, 12), latitude=24.57, longitude=46.53,
        altitude_deg=45.0, azimuth_deg=azimuth_deg, zenith_deg=45.0,
        is_daytime=True,
    )


def test_southern_building_casts_northward_shadow_on_road():
    road = LineString([(46.5300, 24.5700), (46.5300, 24.5710)])
    # A 20m building just south of the road; a south sun casts a north shadow.
    building = Polygon([
        (46.52995, 24.56992), (46.53005, 24.56992),
        (46.53005, 24.56998), (46.52995, 24.56998),
    ])
    result = ShadowEstimator().estimate_shade_from_footprints(
        road, 6.0, _sun(180.0), [{"geometry": building, "height_m": 20.0}],
    )
    assert result.shade_fraction > 0
    assert result.dominant_source == "building"


def test_shadow_direction_changes_with_sun_azimuth():
    road = LineString([(46.5300, 24.5700), (46.5300, 24.5710)])
    building = Polygon([
        (46.52995, 24.56992), (46.53005, 24.56992),
        (46.53005, 24.56998), (46.52995, 24.56998),
    ])
    result = ShadowEstimator().estimate_shade_from_footprints(
        road, 6.0, _sun(0.0), [{"geometry": building, "height_m": 20.0}],
    )
    assert result.shade_fraction == 0
