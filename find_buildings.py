import geopandas as gpd
import warnings

warnings.filterwarnings("ignore")

def find_buildings_in_gpkg(file_path):
    print("=" * 60)
    print("🏢 🏙️ Extracting Building Data & Heights from Map 🏙️ 🏢")
    print("=" * 60)
    
    try:
        # Buildings are represented as polygons
        print("⏳ Loading polygon layer (this may take a few seconds)...")
        gdf_polys = gpd.read_file(file_path, layer='multipolygons')
        
        # Check if building column exists
        if 'building' in gdf_polys.columns:
            buildings = gdf_polys[gdf_polys['building'].notna() & (gdf_polys['building'] != 'no')]
            total_buildings = len(buildings)
            print(f"\n✅ Found {total_buildings} buildings in the selected area.")
            
            # Check for heights / levels
            buildings_with_height = 0
            buildings_with_levels = 0
            
            if 'height' in buildings.columns:
                has_height = buildings['height'].notna()
                buildings_with_height = has_height.sum()
                
            if 'building:levels' in buildings.columns:
                has_levels = buildings['building:levels'].notna()
                buildings_with_levels = has_levels.sum()
                
            print(f"\n📏 Height Data Availability:")
            print(f"   - Buildings with explicit 'height' tag: {buildings_with_height}")
            print(f"   - Buildings with 'building:levels' (number of floors) tag: {buildings_with_levels}")
            
            if buildings_with_height == 0 and buildings_with_levels == 0:
                print("\n⚠️ Note: No explicit height or floors data was mapped in OSM for this specific area.")
                print("   Solution: For route optimization and shadow calculation, you normally")
                print("   assume an average default height (e.g., 2 floors = 7 meters) for all buildings.")
            else:
                print("\n💡 Tip: You can calculate the estimated height from levels by multiplying by 3.5 (e.g. 1 layer = 3.5 meters).")
                
        else:
            print(f"\n❌ No 'building' tag found in the polygons layer.")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    GPKG_PATH = r"حي العليا\planet_46.66,24.685_46.695,24.72.gpkg"
    find_buildings_in_gpkg(GPKG_PATH)
