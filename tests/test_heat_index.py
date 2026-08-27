"""Tests for the Heat Index Calculator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from src.feature_engineering.heat_index_calculator import HeatIndexCalculator


class TestHeatIndex:
    """Test the NWS Rothfusz Heat Index calculation."""

    def setup_method(self):
        self.calc = HeatIndexCalculator()

    def test_low_temperature_returns_reasonable_value(self):
        """Below 27°C (80°F), heat index should be close to air temp."""
        hi = self.calc.heat_index(25.0, 50.0)
        assert 20.0 < hi < 30.0

    def test_high_temperature_high_humidity(self):
        """High temp + high humidity should produce elevated heat index."""
        hi = self.calc.heat_index(40.0, 80.0)
        assert hi > 45.0

    def test_heat_index_increases_with_humidity(self):
        """Higher humidity → higher heat index."""
        hi_low = self.calc.heat_index(35.0, 30.0)
        hi_high = self.calc.heat_index(35.0, 80.0)
        assert hi_high > hi_low

    def test_heat_index_increases_with_temperature(self):
        """Higher temperature → higher heat index."""
        hi_low = self.calc.heat_index(30.0, 50.0)
        hi_high = self.calc.heat_index(45.0, 50.0)
        assert hi_high > hi_low


class TestWBGT:
    """Test the simplified WBGT calculation."""

    def setup_method(self):
        self.calc = HeatIndexCalculator()

    def test_wbgt_basic(self):
        """WBGT should return a reasonable value."""
        wbgt = self.calc.wbgt_simplified(35.0, 60.0, 500.0, 2.0)
        assert 20.0 < wbgt < 50.0

    def test_wbgt_increases_with_radiation(self):
        """Higher solar radiation → higher WBGT."""
        wbgt_shade = self.calc.wbgt_simplified(35.0, 60.0, 0.0, 2.0)
        wbgt_sun = self.calc.wbgt_simplified(35.0, 60.0, 800.0, 2.0)
        assert wbgt_sun > wbgt_shade

    def test_wbgt_wind_cooling(self):
        """Higher wind → lower WBGT (convective cooling)."""
        wbgt_calm = self.calc.wbgt_simplified(40.0, 50.0, 500.0, 0.5)
        wbgt_windy = self.calc.wbgt_simplified(40.0, 50.0, 500.0, 5.0)
        assert wbgt_windy < wbgt_calm


class TestApparentTemperature:
    """Test apparent temperature calculation."""

    def setup_method(self):
        self.calc = HeatIndexCalculator()

    def test_apparent_temp_basic(self):
        at = self.calc.apparent_temperature(35.0, 50.0, 2.0)
        assert 25.0 < at < 45.0

    def test_wind_cools_apparent_temp(self):
        at_calm = self.calc.apparent_temperature(35.0, 50.0, 1.0)
        at_windy = self.calc.apparent_temperature(35.0, 50.0, 10.0)
        assert at_windy < at_calm


class TestThermalIndices:
    """Test the combined calculation."""

    def setup_method(self):
        self.calc = HeatIndexCalculator()

    def test_calculate_all_returns_all_fields(self):
        result = self.calc.calculate_all(35.0, 60.0, 10.0, 500.0)
        assert result.heat_index_c > 0
        assert result.wbgt_c > 0
        assert result.apparent_temperature_c is not None
        assert result.thermal_stress_category in ["comfortable", "caution", "danger", "extreme"]

    def test_extreme_conditions(self):
        result = self.calc.calculate_all(50.0, 80.0, 5.0, 900.0)
        assert result.thermal_stress_category in ["danger", "extreme"]


class TestBatchCalculation:
    """Test vectorized calculations."""

    def test_batch_heat_index(self):
        temps = np.array([25.0, 35.0, 45.0])
        humidities = np.array([50.0, 60.0, 70.0])
        result = HeatIndexCalculator.heat_index_batch(temps, humidities)
        assert len(result) == 3
        assert result[2] > result[1] > result[0]
