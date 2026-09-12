import geopandas as gpd
import warnings

warnings.filterwarnings("ignore")

def find_trees_in_gpkg(file_path):
    print("=" * 60)
    print("🌳 🌴 Extracting Tree Data from Map 🌴 🌳")
    print("=" * 60)
    
    try:
        # Search in points layer (single trees)
        gdf_points = gpd.read_file(file_path, layer='points')
        if 'natural' in gdf_points.columns:
            trees_points = gdf_points[gdf_points['natural'] == 'tree']
            print(f"\n1️⃣ Single Trees (Points):")
            print(f"   ✅ Found {len(trees_points)} single trees.")
            if len(trees_points) > 0:
                print("   📍 Sample Coordinates (first 3 trees):")
                # Print x and y coordinates
                for idx, row in trees_points.head(3).iterrows():
                    print(f"      - Longitude: {row.geometry.x:.5f}, Latitude: {row.geometry.y:.5f}")
        else:
            print("\n1️⃣ Single Trees: No 'natural' column found in points.")

        # Search in polygons layer (forests, parks, woods)
        gdf_polys = gpd.read_file(file_path, layer='multipolygons')
        
        # Tree areas
        tree_areas = 0
        if 'natural' in gdf_polys.columns:
            woods = gdf_polys[gdf_polys['natural'] == 'wood']
            tree_areas += len(woods)
        
        if 'landuse' in gdf_polys.columns:
            forests = gdf_polys[gdf_polys['landuse'] == 'forest']
            tree_areas += len(forests)

        print(f"\n2️⃣ Forests & Wooded Areas (Polygons):")
        print(f"   ✅ Found {tree_areas} shaded tree/forest areas.")

        # Parks
        if 'leisure' in gdf_polys.columns:
            parks = gdf_polys[gdf_polys['leisure'] == 'park']
            print(f"\n3️⃣ Public Parks (Usually contain dense trees):")
            print(f"   ✅ Found {len(parks)} parks drawn as polygons.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    GPKG_PATH = r"planet_46.48,24.5401_46.5789,24.6004-geopackage\planet_46.48,24.5401_46.5789,24.6004.gpkg"
    find_trees_in_gpkg(GPKG_PATH)
