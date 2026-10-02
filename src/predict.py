import yaml
import rasterio
import numpy as np
import joblib
import glob
import os

# 1. Config laden
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

raw_dir = config['paths']['raw_satellite_dir']
model_path = config['paths']['model_output']
output_path = config['paths']['prediction_output']
bands = config['satellite']['bands']

print("Lade trainiertes Modell...")
rf = joblib.load(model_path)

print("Lade Sentinel-2 Bänder für das gesamte Bild...")
img_data = {}
profile = None

# Alle 4 Bänder einlesen
for band in bands:
    band_pattern = os.path.join(raw_dir, f"*{band}*.jp2")
    band_path = glob.glob(band_pattern)[0]
    with rasterio.open(band_path) as src:
        # Als Float einlesen für NDVI berechnungen
        img_data[band] = src.read(1).astype(np.float32) 
        if profile is None:
            profile = src.profile # Metadaten für Export speichern

# NDVI für das gesamte Bild berechnen
features = bands.copy()
if config['satellite']['calculate_ndvi']:
    print("Berechne NDVI für das gesamte Bild...")
    nir = img_data['B08']
    red = img_data['B04']
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))
    img_data['NDVI'] = ndvi
    features.append('NDVI')

print("Formatiere Daten für das Modell um...")
# Umwandlung der 2D Karte in Liste
height, width = img_data[bands[0]].shape
n_pixels = height * width

# Leere Matrix vorbereiten: Zeilen = Pixelanzahl, Spalten = Anzahl der Features (5)
X_full = np.zeros((n_pixels, len(features)), dtype=np.float32)

for i, feature in enumerate(features):
    X_full[:, i] = img_data[feature].flatten()

# Arbeitsspeicher entlasten
del img_data

print("Starte Klassifizierung (das kann 1-3 Minuten dauern)...")
# Pixelvorhersage
y_pred = rf.predict(X_full)

print("Formatiere Ergebnis zurück in ein Bild...")
# Liste wieder in 2D format zurückversetzen
forest_mask = y_pred.reshape((height, width))

print(f"Speichere das finale Ergebnis als GeoTIFF unter {output_path}...")
# Profile für das Ausgabe-Bild anpassen (1 & 0)
profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress='lzw' # LZW-Komprimierung hält die Dateigröße klein
)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
with rasterio.open(output_path, 'w', **profile) as dst:
    dst.write(forest_mask.astype(rasterio.uint8), 1)

print("\n>>> FERTIG! Der ML-Algorithmus hat die Sentinel-2 Szene erfolgreich klassifiziert. <<<")