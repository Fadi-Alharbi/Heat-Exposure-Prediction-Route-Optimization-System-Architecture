"""Tests for the Solar Calculator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import datetime as dt
import pytest
from src.feature_engineering.solar_calculator import SolarCalculator


class TestSolarCalculator:
    """Test solar position calculations."""

    def setup_method(self):
        self.calc = SolarCalculator()

    def test_midday_sun_is_high(self):
        """At solar noon in Riyadh, sun altitude should be high."""
        # Riyadh, August noon UTC+3
        ts = dt.datetime(2026, 8, 15, 12, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=3)))
        pos = self.calc.get_solar_position(24.7136, 46.6753, ts)
        assert pos.altitude_deg > 50  # sun should be high at noon in August
        assert pos.is_daytime

    def test_midnight_sun_is_below_horizon(self):
        """At midnight, sun altitude should be 0 or negative."""
        ts = dt.datetime(2026, 8, 15, 0, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=3)))
        pos = self.calc.get_solar_position(24.7136, 46.6753, ts)
        assert pos.altitude_deg <= 5  # should be very low or below horizon

    def test_azimuth_range(self):
        """Azimuth should be between 0 and 360 degrees."""
        ts = dt.datetime(2026, 8, 15, 10, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=3)))
        pos = self.calc.get_solar_position(24.7136, 46.6753, ts)
        assert 0 <= pos.azimuth_deg <= 360

    def test_zenith_altitude_complement(self):
        """Zenith + altitude should ≈ 90 (when sun is up)."""
        ts = dt.datetime(2026, 8, 15, 12, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=3)))
        pos = self.calc.get_solar_position(24.7136, 46.6753, ts)
        if pos.is_daytime:
            assert abs(pos.altitude_deg + pos.zenith_deg - 90) < 1.0

    def test_hourly_profile_length(self):
        """Hourly profile should have 24 rows."""
        df = self.calc.get_solar_positions_hourly(
            24.7136, 46.6753,
            dt.date(2026, 8, 15),
            timezone="Asia/Riyadh",
        )
        assert len(df) == 24
        assert "altitude_deg" in df.columns
        assert "azimuth_deg" in df.columns
