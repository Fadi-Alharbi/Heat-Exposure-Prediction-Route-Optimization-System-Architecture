import geopandas as gpd
import folium
import warnings
import os

warnings.filterwarnings("ignore")

def create_buildings_map(file_path):
    print("⏳ Loading buildings from GeoPackage...")
    # Load polygons
    gdf_polys = gpd.read_file(file_path, layer='multipolygons')
    
    # Filter for buildings
    if 'building' in gdf_polys.columns:
        buildings = gdf_polys[gdf_polys['building'].notna() & (gdf_polys['building'] != 'no')]
        print(f"✅ Found {len(buildings)} buildings.")
    else:
        print("❌ No buildings found!")
        return

    # Calculate center of the area to center the map
    center_lat = buildings.geometry.centroid.y.mean()
    center_lon = buildings.geometry.centroid.x.mean()
    
    print("🗺️ Generating Interactive Map...")
    # Create Folium Map centered on the buildings
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB positron')
    
    # Add buildings to map
    for _, row in buildings.iterrows():
        # Get building height or assume default
        height = row.get('height', 'Unknown')
        levels = row.get('building:levels', 'Unknown')
        
        popup_info = f"<b>Building Info</b><br>Levels: {levels}<br>Height: {height}"
        
        # Convert geometry to GeoJSON and add to folium map
        geo_j = folium.GeoJson(
            data=row['geometry'],
            style_function=lambda x: {
                'fillColor': '#ff0000', # Red color for visibility
                'color': '#8b0000',     # Border color
                'weight': 1,
                'fillOpacity': 0.6
            },
            popup=folium.Popup(popup_info, max_width=200)
        )
        geo_j.add_to(m)
        
    # Save the map
    output_file = "buildings_map.html"
    m.save(output_file)
    print(f"🎉 Success! The map has been saved as: {os.path.abspath(output_file)}")
    print(f"👉 Double-click the file to open it in your web browser.")

if __name__ == "__main__":
    GPKG_PATH = r"planet_46.48,24.5401_46.5789,24.6004-geopackage\planet_46.48,24.5401_46.5789,24.6004.gpkg"
    create_buildings_map(GPKG_PATH)
