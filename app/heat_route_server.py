"""
Heat-Aware Route Server
───────────────────────
Standalone FastAPI app that:
  1. Loads the GeoPackage road network + buildings + trees
  2. Accepts origin/destination + time from the frontend
  3. Builds a heat-weighted graph using solar/shadow/surface analysis
  4. Finds 3 routes: Fastest, Coolest, Balanced
  5. Returns JSON routes + stats to the interactive map UI
"""

import sys
import os
import datetime as dt
import math
import json
import traceback

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import networkx as nx
import numpy as np

from config.settings import settings
from src.data_ingestion.osm_fetcher import OSMFetcher
from src.data_ingestion.weather_client import WeatherClient, WeatherSnapshot
from src.optimization.graph_builder import GraphBuilder
from src.optimization.path_finder import PathFinder

# ── App ──────────────────────────────────────────────────────────
app = FastAPI(title="Heat Route Optimizer", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper: convert numpy types to native Python ────────────────
def to_native(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# ── Load map data once ───────────────────────────────────────────
GPKG_PATH = os.path.join(
    PROJECT_ROOT,
    "planet_46.48,24.5401_46.5789,24.6004-geopackage",
    "planet_46.48,24.5401_46.5789,24.6004.gpkg",
)

print("⏳ Loading GeoPackage map data...")
fetcher = OSMFetcher()
MAP_DATA = fetcher.fetch_from_geopackage(GPKG_PATH)
print(f"✅ Map loaded: {MAP_DATA.road_graph.number_of_nodes()} nodes, {MAP_DATA.road_graph.number_of_edges()} edges")

# Precompute the largest strongly connected component
print("⏳ Computing largest connected component...")
_components = list(nx.strongly_connected_components(MAP_DATA.road_graph))
_largest_cc = max(_components, key=len)
BASE_GRAPH = MAP_DATA.road_graph.subgraph(_largest_cc).copy()
print(f"✅ Connected component: {BASE_GRAPH.number_of_nodes()} nodes, {BASE_GRAPH.number_of_edges()} edges")

# Bounding box from the BBBike extract
MAP_BOUNDS = {
    "south": 24.5401,
    "north": 24.6004,
    "west": 46.48,
    "east": 46.5789,
    "center_lat": (24.5401 + 24.6004) / 2,
    "center_lon": (46.48 + 46.5789) / 2,
}

print(f"✅ Map center: {MAP_BOUNDS['center_lat']:.4f}, {MAP_BOUNDS['center_lon']:.4f}")


# ── Request model ────────────────────────────────────────────────

class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    trip_hour: int = 12
    trip_minute: int = 0
    trip_date: str = ""
    speed_kmh: float = 15.0


# ── Helper ───────────────────────────────────────────────────────

def get_closest_node(lon, lat, G):
    best_n = None
    best_d = float('inf')
    for n, data in G.nodes(data=True):
        nx_val = float(data['x'])
        ny_val = float(data['y'])
        d = (nx_val - lon) ** 2 + (ny_val - lat) ** 2
        if d < best_d:
            best_d = d
            best_n = n
    return best_n


def get_route_area_graph(origin_lat, origin_lon, dest_lat, dest_lon, buffer_deg=0.004):
    """Return the local connected BBBike subgraph around one requested trip."""
    south = min(origin_lat, dest_lat) - buffer_deg
    north = max(origin_lat, dest_lat) + buffer_deg
    west = min(origin_lon, dest_lon) - buffer_deg
    east = max(origin_lon, dest_lon) + buffer_deg
    nodes = [
        node for node, attrs in BASE_GRAPH.nodes(data=True)
        if south <= attrs["y"] <= north and west <= attrs["x"] <= east
    ]
    local_graph = BASE_GRAPH.subgraph(nodes).copy()
    if not local_graph:
        return local_graph
    components = list(nx.strongly_connected_components(local_graph))
    return local_graph.subgraph(max(components, key=len)).copy() if components else local_graph


# ── API Endpoints ────────────────────────────────────────────────

@app.get("/api/map-info")
def get_map_info():
    """Return map bounds and center for initial view."""
    return JSONResponse(content=to_native(MAP_BOUNDS))


@app.get("/api/road-network")
def get_road_network():
    """Return the road network as GeoJSON for overlay."""
    features = []
    seen = set()
    for u, v, data in BASE_GRAPH.edges(data=True):
        edge_key = (u, v)
        if edge_key in seen:
            continue
        seen.add(edge_key)
        u_data = BASE_GRAPH.nodes[u]
        v_data = BASE_GRAPH.nodes[v]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [float(u_data['x']), float(u_data['y'])],
                    [float(v_data['x']), float(v_data['y'])]
                ]
            },
            "properties": {
                "highway": str(data.get("highway", "unknown")),
                "name": str(data.get("name", "")),
            }
        })
    return JSONResponse(content=to_native({
        "type": "FeatureCollection",
        "features": features
    }))


@app.post("/api/find-routes")
def find_routes(req: RouteRequest):
    """Find Fastest, Coolest, and Balanced routes."""
    try:
        # Parse date/time
        if req.trip_date:
            trip_date = dt.date.fromisoformat(req.trip_date)
        else:
            trip_date = dt.date.today()
        trip_time = dt.datetime.combine(trip_date, dt.time(req.trip_hour, req.trip_minute))

        # Fetch weather
        client = WeatherClient()
        try:
            weather = client.get_current(MAP_BOUNDS["center_lat"], MAP_BOUNDS["center_lon"])
        except Exception as we:
            print(f"⚠️ Weather fetch failed: {we}")
            weather = None

        if weather is None:
            weather = WeatherSnapshot(
                timestamp=dt.datetime.now(),
                latitude=MAP_BOUNDS["center_lat"],
                longitude=MAP_BOUNDS["center_lon"],
                temperature_c=42.0,
                relative_humidity_pct=15.0,
                wind_speed_kmh=12.0,
                direct_radiation_wm2=750.0,
                diffuse_radiation_wm2=120.0,
                cloud_cover_pct=5.0,
            )

        if not (MAP_BOUNDS["south"] <= req.origin_lat <= MAP_BOUNDS["north"]
                and MAP_BOUNDS["west"] <= req.origin_lon <= MAP_BOUNDS["east"]
                and MAP_BOUNDS["south"] <= req.dest_lat <= MAP_BOUNDS["north"]
                and MAP_BOUNDS["west"] <= req.dest_lon <= MAP_BOUNDS["east"]):
            return JSONResponse(content={
                "success": False, "routes": [], "weather": {}, "solar": {},
                "error": "Choose origin and destination inside the BBBike map boundary."
            })

        # Weight only the connected local corridor for this trip.  The map
        # still comes entirely from the bundled BBBike GeoPackage, while
        # avoiding an expensive recomputation of unrelated streets.
        route_graph = get_route_area_graph(
            req.origin_lat, req.origin_lon, req.dest_lat, req.dest_lon,
        )
        if route_graph.number_of_edges() == 0:
            return JSONResponse(content={
                "success": False, "routes": [], "weather": {}, "solar": {},
                "error": "No connected BBBike streets were found near this trip."
            })

        # Build weighted graph
        builder = GraphBuilder()
        weighted_graph = builder.build_weighted_graph(
            graph=route_graph,
            weather=weather,
            trip_time=trip_time,
            user_speed_kmh=req.speed_kmh,
            buildings_gdf=MAP_DATA.buildings_gdf,
            trees_gdf=MAP_DATA.trees_gdf,
        )

        # Find closest nodes
        u_node = get_closest_node(req.origin_lon, req.origin_lat, weighted_graph)
        v_node = get_closest_node(req.dest_lon, req.dest_lat, weighted_graph)

        if u_node is None or v_node is None:
            return JSONResponse(content={
                "success": False, "routes": [], "weather": {}, "solar": {},
                "error": "Could not find nodes near the selected points."
            })

        if u_node == v_node:
            return JSONResponse(content={
                "success": False, "routes": [], "weather": {}, "solar": {},
                "error": "Origin and destination are too close. Please select different points."
            })

        # Find routes
        finder = PathFinder()
        routes = finder.compare_routes(weighted_graph, u_node, v_node)

        if not routes:
            return JSONResponse(content={
                "success": False, "routes": [], "weather": {}, "solar": {},
                "error": "No path found between these points. Try selecting points closer to roads."
            })

        # Build response with native Python types
        colors = {"Fastest": "#FF5252", "Coolest": "#00E676", "Balanced": "#FFD740"}
        route_infos = []
        for r in routes:
            coords = []
            for node_id in r.path_nodes:
                n_data = weighted_graph.nodes[node_id]
                coords.append([float(n_data['y']), float(n_data['x'])])

            route_infos.append({
                "label": str(r.label),
                "route_id": str(r.route_id),
                "is_recommended": bool(r.is_recommended),
                "total_distance_m": float(r.total_distance_m),
                "total_time_min": float(r.total_time_min),
                "avg_heat_exposure": float(r.avg_heat_exposure),
                "max_heat_exposure": float(r.max_heat_exposure),
                "cumulative_heat_exposure": float(r.cumulative_heat_exposure),
                "avg_shade_fraction": float(r.avg_shade_fraction),
                "combined_cost": float(r.combined_cost),
                "coordinates": coords,
                "color": colors.get(r.label, "#FFFFFF"),
                "segments": [{
                    "heat_score": float(s.get("heat_score", 0)),
                    "shade_fraction": float(s.get("shade_fraction", 0)),
                    "surface_type": str(s.get("surface_type", "unknown")),
                } for s in r.segments[:20]]
            })

        # Solar info
        from src.feature_engineering.solar_calculator import SolarCalculator
        solar_calc = SolarCalculator()
        solar = solar_calc.get_solar_position(
            MAP_BOUNDS["center_lat"], MAP_BOUNDS["center_lon"], trip_time
        )

        response = {
            "success": True,
            "routes": route_infos,
            "weather": {
                "temperature_c": float(weather.temperature_c),
                "humidity_pct": float(weather.relative_humidity_pct),
                "wind_kmh": float(weather.wind_speed_kmh),
                "radiation_wm2": float(weather.total_radiation_wm2),
                "cloud_cover_pct": float(weather.cloud_cover_pct),
            },
            "solar": {
                "altitude_deg": round(float(solar.altitude_deg), 1),
                "azimuth_deg": round(float(solar.azimuth_deg), 1),
                "is_daytime": bool(solar.is_daytime),
                "sunrise": solar.sunrise.strftime("%H:%M") if solar.sunrise else "N/A",
                "sunset": solar.sunset.strftime("%H:%M") if solar.sunset else "N/A",
            },
            "error": ""
        }

        return JSONResponse(content=response)

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={
            "success": False, "routes": [], "weather": {}, "solar": {},
            "error": str(e)
        })


# ── Serve Frontend ───────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)
