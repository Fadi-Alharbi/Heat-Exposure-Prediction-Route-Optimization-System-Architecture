import geopandas as gpd
import fiona
import warnings
import pandas as pd
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def export_geopackage_to_csv(gpkg_path):
    print("\n" + "="*70)
    print("🌍 🗺️  GeoPackage Map Exporter  🗺️ 🌍")
    print("="*70)
    
    # Verify layers
    layers = fiona.listlayers(gpkg_path)
    
    #######################################################
    # Part 1: Exporting Points (Places & POIs)
    #######################################################
    if 'points' in layers:
        print("\n⏳ Extracting Places & Points of Interest ('points' layer)...")
        
        gdf_points = gpd.read_file(gpkg_path, layer='points')
        
        if 'name' in gdf_points.columns:
            named_points = gdf_points[gdf_points['name'].notna()].copy()
            
            # Select relevant columns for the user
            export_points = named_points[['name', 'geometry']].head(50).copy()
            
            # Save to CSV with utf-8-sig so Excel reads Arabic correctly
            export_points.to_csv('map_places_output.csv', index=False, encoding='utf-8-sig')
            print(f"✔️ Successfully saved 50 places to 'map_places_output.csv'!")
    
    #######################################################
    # Part 2: Exporting Lines (Street Network)
    #######################################################
    if 'lines' in layers:
        print("\n⏳ Extracting Street Network ('lines' layer)...")
        
        gdf_lines = gpd.read_file(gpkg_path, layer='lines')
        
        if 'name' in gdf_lines.columns and 'highway' in gdf_lines.columns:
            named_lines = gdf_lines[gdf_lines['name'].notna()].copy()
            
            # Fill missing highway types
            named_lines['highway_type'] = named_lines['highway'].fillna('Unknown')
            
            # Select relevant columns 
            export_lines = named_lines[['name', 'highway_type', 'geometry']].head(50).copy()
            
            # Save to CSV
            export_lines.to_csv('map_streets_output.csv', index=False, encoding='utf-8-sig')
            print(f"✔️ Successfully saved 50 streets to 'map_streets_output.csv'!")

    print("\n" + "="*70)
    print("✅ All data extracted successfully! Go check the new CSV files in your folder.")
    print("="*70 + "\n")


if __name__ == "__main__":
    file_path = "planet_46.48,24.5401_46.5789,24.6004-geopackage/planet_46.48,24.5401_46.5789,24.6004.gpkg"
    export_geopackage_to_csv(file_path)
