import datetime as dt
import networkx as nx
from src.data_ingestion.osm_fetcher import OSMFetcher
from src.data_ingestion.weather_client import WeatherSnapshot
from src.optimization.graph_builder import GraphBuilder
from src.optimization.path_finder import PathFinder

def main():
    print("1. Loading geographic data from local GeoPackage...")
    fetcher = OSMFetcher()
    gpkg_path = "planet_46.48,24.5401_46.5789,24.6004-geopackage/planet_46.48,24.5401_46.5789,24.6004.gpkg"
    data = fetcher.fetch_from_geopackage(gpkg_path)
    print(f"   -> Found {len(data.road_graph.nodes)} nodes and {len(data.road_graph.edges)} edges.")

    print("\n2. Applying ML & Heat Models (Simulating 42°C at 2:00 PM)...")
    weather = WeatherSnapshot(
        latitude=data.center[0],
        longitude=data.center[1],
        temperature_c=42.0,
        relative_humidity_pct=15.0,
        wind_speed_kmh=8.0,
        direct_radiation_wm2=800.0,
        diffuse_radiation_wm2=100.0,
        cloud_cover_pct=5.0,
        timestamp=dt.datetime.now()
    )
    
    trip_time = dt.datetime(2026, 9, 7, 14, 0, 0)
    
    builder = GraphBuilder()
    graph = builder.build_weighted_graph(
        graph=data.road_graph,
        weather=weather,
        trip_time=trip_time,
        buildings_gdf=data.buildings_gdf,
        trees_gdf=data.trees_gdf
    )
    
    print("\n3. Finding Routes (Fastest vs Coolest)...")
    # Get the largest connected component to ensure a path exists
    largest_cc = max(nx.weakly_connected_components(graph), key=len)
    cc_nodes = list(largest_cc)
    
    # Select two distant points in the neighborhood for a good demo
    origin = cc_nodes[0]
    destination = cc_nodes[int(len(cc_nodes)/2)]
    
    finder = PathFinder()
    routes = finder.compare_routes(graph, origin, destination)

    print("\n" + "="*50)
    print("🚗 N E I G H B O R H O O D   R O U T I N G   🚗")
    print("="*50)
    
    if not routes:
        print("No paths could be found between the selected points.")
        return

    for r in routes:
        rec = "⭐ RECOMMENDED" if r.is_recommended else ""
        print(f"\n[{r.label} Route] {rec}")
        print(f"⏱️ Duration: {r.total_time_min} mins")
        print(f"🌡️ Heat Exposure Level: {r.cumulative_heat_exposure}")
        print(f"🌳 Average Shade Coverage: {r.avg_shade_fraction*100:.1f}%")
        print(f"🛣️ Distance: {r.total_distance_m} meters")
        
if __name__ == "__main__":
    main()
