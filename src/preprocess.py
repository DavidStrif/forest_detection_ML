import yaml
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pandas as pd
import numpy as np
import glob  
import os

# 1. Config laden
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

raw_dir = config['paths']['raw_satellite_dir']
gpkg_path = config['paths']['training_polygons']
layer_name = config['geo']['gpkg_layer']
label_col = config['geo']['label_column']
bands = config['satellite']['bands']

print("Lade GeoPackage...")
# 2. GeoPackage laden
gdf = gpd.read_file(gpkg_path, layer=layer_name)

# Dateien mit Bandnamen im Namen suchen
ref_raster_pattern = os.path.join(raw_dir, f"*{bands[0]}*.jp2")
found_files = glob.glob(ref_raster_pattern)

if len(found_files) == 0:
    raise FileNotFoundError(f"Konnte keine Datei für Band {bands[0]} in {raw_dir} finden!")
ref_raster_path = found_files[0]

with rasterio.open(ref_raster_path) as src:
    print(f"Satellitenbild CRS: {src.crs}")
    gdf = gdf.to_crs(src.crs)

all_pixels = []
print("Extrahiere Pixel aus den Satellitenbildern. Das kann einen Moment dauern...")

# 4. Durch jedes Polygon im GeoPackage iterieren
for idx, row in gdf.iterrows():
    geom = [row.geometry]
    label = row[label_col]
    
    poly_pixels = {'label': label}
    
    # Durch alle 4 Bänder gehen
    for band in bands:
        band_pattern = os.path.join(raw_dir, f"*{band}*.jp2")
        band_path = glob.glob(band_pattern)[0]
        
        with rasterio.open(band_path) as src:
            out_image, out_transform = mask(src, geom, crop=True)
            # 2D bild wird zu Liste geglättet (alle Werte behalten, auch Nullen)
            poly_pixels[band] = out_image[0].flatten()
            
    # NDVI berechnen: 
    if config['satellite']['calculate_ndvi']:
        nir = poly_pixels['B08'].astype(float)
        red = poly_pixels['B04'].astype(float)
        
        # RuntimeWarning bei Division durch Null ignorieren
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
        poly_pixels['NDVI'] = ndvi
        
    # Tabelle für dieses Polygon erstellen
    df_poly = pd.DataFrame(poly_pixels)
    
    # Wenn ein Pixel in B02 (oder einem anderen Band) 0 ist, stammt es vom schwarzen Rand der Maske.
    df_poly = df_poly[df_poly['B02'] > 0]
    
    all_pixels.append(df_poly)

# 5. Tabellen zusammenfügen
final_df = pd.concat(all_pixels, ignore_index=True)

# 6. Speichern
output_path = "data/interim/training_data.csv"
final_df.to_csv(output_path, index=False)

print(f"\nErfolgreich! {len(final_df)} Pixel für das Training extrahiert.")