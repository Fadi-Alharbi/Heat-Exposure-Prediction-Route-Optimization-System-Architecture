"""
Surface Type Classifier
───────────────────────
Classifies road surface types from OpenStreetMap tags and assigns
thermal properties (heat absorption factor, albedo, emissivity).
"""

from __future__ import annotations

from dataclasses import dataclass

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import settings


@dataclass
class SurfaceProperties:
    """Thermal and physical properties of a road surface."""
    surface_type: str
    heat_factor: float      # 0-1 relative heat absorption
    albedo: float           # solar reflectance (0-1, higher = cooler)
    emissivity: float       # thermal emissivity (0-1)
    roughness: str          # "smooth" | "medium" | "rough"


class SurfaceClassifier:
    """
    Classifies road surfaces from OSM tags and returns thermal properties.

    OSM uses the 'surface' tag on ways. When not present, we infer from
    the 'highway' tag (motorway → asphalt, track → compacted, etc.).
    """

    # Full surface property catalog
    SURFACE_CATALOG: dict[str, SurfaceProperties] = {
        "asphalt": SurfaceProperties("asphalt", 0.95, 0.12, 0.93, "smooth"),
        "concrete": SurfaceProperties("concrete", 0.85, 0.30, 0.92, "smooth"),
        "concrete:plates": SurfaceProperties("concrete", 0.85, 0.30, 0.92, "smooth"),
        "concrete:lanes": SurfaceProperties("concrete", 0.85, 0.30, 0.92, "smooth"),
        "paving_stones": SurfaceProperties("paving_stones", 0.80, 0.25, 0.90, "medium"),
        "sett": SurfaceProperties("paving_stones", 0.80, 0.25, 0.90, "medium"),
        "cobblestone": SurfaceProperties("paving_stones", 0.80, 0.25, 0.90, "rough"),
        "compacted": SurfaceProperties("compacted", 0.70, 0.35, 0.85, "medium"),
        "fine_gravel": SurfaceProperties("gravel", 0.60, 0.40, 0.80, "medium"),
        "gravel": SurfaceProperties("gravel", 0.60, 0.40, 0.80, "rough"),
        "pebblestone": SurfaceProperties("gravel", 0.60, 0.40, 0.80, "rough"),
        "sand": SurfaceProperties("sand", 0.55, 0.45, 0.76, "rough"),
        "dirt": SurfaceProperties("dirt", 0.50, 0.30, 0.90, "rough"),
        "earth": SurfaceProperties("dirt", 0.50, 0.30, 0.90, "rough"),
        "mud": SurfaceProperties("dirt", 0.50, 0.20, 0.95, "rough"),
        "grass": SurfaceProperties("grass", 0.30, 0.25, 0.95, "rough"),
        "grass_paver": SurfaceProperties("grass", 0.40, 0.22, 0.93, "medium"),
        "ground": SurfaceProperties("ground", 0.50, 0.30, 0.90, "rough"),
        "unpaved": SurfaceProperties("ground", 0.50, 0.30, 0.90, "rough"),
        "metal": SurfaceProperties("asphalt", 0.90, 0.15, 0.85, "smooth"),
        "wood": SurfaceProperties("ground", 0.45, 0.35, 0.90, "medium"),
        "tartan": SurfaceProperties("asphalt", 0.88, 0.15, 0.92, "smooth"),
    }

    # Highway-type to default surface mapping
    HIGHWAY_DEFAULTS: dict[str, str] = {
        "motorway": "asphalt",
        "motorway_link": "asphalt",
        "trunk": "asphalt",
        "trunk_link": "asphalt",
        "primary": "asphalt",
        "primary_link": "asphalt",
        "secondary": "asphalt",
        "secondary_link": "asphalt",
        "tertiary": "asphalt",
        "tertiary_link": "asphalt",
        "residential": "asphalt",
        "service": "asphalt",
        "unclassified": "asphalt",
        "living_street": "paving_stones",
        "pedestrian": "paving_stones",
        "footway": "paving_stones",
        "cycleway": "asphalt",
        "path": "compacted",
        "track": "compacted",
        "bridleway": "ground",
    }

    def classify(
        self,
        surface_tag: str | None = None,
        highway_tag: str | None = None,
    ) -> SurfaceProperties:
        """
        Return thermal properties for a given surface/highway tag.

        Parameters
        ----------
        surface_tag : str or None
            OSM 'surface' tag value (e.g., "asphalt", "gravel").
        highway_tag : str or None
            OSM 'highway' tag value (e.g., "residential", "footway").
        """
        # Try explicit surface tag first
        if surface_tag:
            key = surface_tag.lower().strip()
            if key in self.SURFACE_CATALOG:
                return self.SURFACE_CATALOG[key]

        # Fall back to highway type
        if highway_tag:
            default_surface = self.HIGHWAY_DEFAULTS.get(
                highway_tag.lower().strip(), "asphalt"
            )
            return self.SURFACE_CATALOG.get(
                default_surface,
                self.SURFACE_CATALOG["asphalt"],
            )

        # Unknown
        return SurfaceProperties("unknown", 0.75, 0.20, 0.90, "medium")

    def get_heat_factor(
        self,
        surface_tag: str | None = None,
        highway_tag: str | None = None,
    ) -> float:
        """Shortcut: return just the heat absorption factor."""
        return self.classify(surface_tag, highway_tag).heat_factor

    def classify_edge(self, edge_data: dict) -> SurfaceProperties:
        """Classify a NetworkX edge using its OSM data dictionary."""
        surface = edge_data.get("surface")
        highway = edge_data.get("highway")
        # highway can be a list in OSMnx
        if isinstance(highway, list):
            highway = highway[0]
        if isinstance(surface, list):
            surface = surface[0]
        return self.classify(surface, highway)
