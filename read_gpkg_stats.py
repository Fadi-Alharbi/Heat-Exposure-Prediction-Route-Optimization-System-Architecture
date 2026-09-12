"""
Comprehensive Data Extraction from GeoPackage
─────────────────────────────────────────────
This script:
1. Automatically reads all layers in the GeoPackage.
2. Extracts detailed statistics for each layer.
3. Saves the data into organized CSV files.
4. Displays a comprehensive summary in the terminal.
"""

import geopandas as gpd
import fiona
import pandas as pd
import warnings
from pathlib import Path

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")


def extract_all_data(gpkg_path: str, output_dir: str = "extracted_data"):
    """
    Extract all data from a GeoPackage and save it to CSV.
    
    Parameters
    ----------
    gpkg_path : str
        Path to the GeoPackage file.
    output_dir : str
        Directory where CSV files will be saved.
    """
    print("\n" + "=" * 80)
    print("🌍 Comprehensive GeoPackage Data Extraction System ")
    print("=" * 80)
    print(f"📂 Target file: {gpkg_path}\n")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    try:
        # 1. Get all layers
        layers = fiona.listlayers(gpkg_path)
        print(f"✅ Found {len(layers)} layers:")
        for i, layer in enumerate(layers, 1):
            print(f"   {i}. {layer}")
        print("-" * 80)

        # 2. Process each layer
        summary = []
        for layer in layers:
            print(f"\n📊 Processing layer: '{layer}' ...")
            
            # Read the layer
            gdf = gpd.read_file(gpkg_path, layer=layer)
            
            # Save as CSV
            csv_filename = output_path / f"{layer}.csv"
            gdf.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"   💾 Saved: {csv_filename}")

            # Layer statistics
            num_rows = len(gdf)
            num_cols = len(gdf.columns)
            
            print(f"   📈 Number of features (rows): {num_rows}")
            print(f"    Number of columns: {num_cols}")
            
            # Determine layer type based on name or content
            layer_type = "Unspecified"
            if layer.lower() in ['lines', 'roads', 'streets']:
                layer_type = "🛣️ Roads and Streets"
            elif layer.lower() in ['multipolygons', 'polygons', 'buildings']:
                layer_type = "🏢 Buildings"
            elif layer.lower() in ['points', 'pois']:
                layer_type = "📍 Points of Interest (POIs) / Trees"
            
            # Add to summary
            summary.append({
                'Layer': layer,
                'Type': layer_type,
                'Features': num_rows,
                'Columns': num_cols,
                'CSV_File': str(csv_filename)
            })

            # Display a sample of the data
            print(f"\n   🔍 Data sample (first 3 rows):")
            sample = gdf.drop(columns=['geometry'], errors='ignore').head(3)
            print(sample.to_string(index=False))
            print("-" * 80)

        # 3. Display final summary
        print("\n" + "=" * 80)
        print(" Final Summary of Extracted Data")
        print("=" * 80)
        
        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        
        # Overall statistics
        total_features = summary_df['Features'].sum()
        print(f"\n🎯 Total number of extracted features: {total_features}")
        print(f"📁 Saved {len(layers)} CSV files in directory: {output_dir}")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # ️ Change this path to match the GeoPackage file path on your machine
    GPKG_FILE = r"planet_46.48,24.5401_46.5789,24.6004-geopackage\planet_46.48,24.5401_46.5789,24.6004.gpkg"
    
    # Check if the file exists
    if not Path(GPKG_FILE).exists():
        print(f"❌ File not found: {GPKG_FILE}")
        print("💡 Please verify the path and try again.")
    else:
        extract_all_data(GPKG_FILE, output_dir="extracted_data")