import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Config laden
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = "data/interim/training_data.csv"
model_output = config['paths']['model_output']
bands = config['satellite']['bands']

print("Lade Trainingsdaten...")
# 2. Daten laden 
df = pd.read_csv(data_path)

# 3. Features (X) und Label (Y) festlegen
# Die Features sind die 4 Bänder + NDVI
features = bands.copy()
if config['satellite']['calculate_ndvi']:
    features.append('NDVI')

X = df[features]
y = df['label'] # Name der Zielspalte im Preprocessing

# 4. Daten aufteilen (z.B. 80% Training, 20% Test) (Wert in config.yaml)
print("Teile Daten in Training und Test auf...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=config['model']['test_size'], 
    random_state=config['model']['random_state']
)

# 5. Random Forest initialisieren und trainieren
print(f"Trainiere Random Forest Modell mit {config['model']['n_estimators']} Bäumen...")
rf = RandomForestClassifier(
    n_estimators=config['model']['n_estimators'],
    max_depth=config['model']['max_depth'],
    random_state=config['model']['random_state'],
    n_jobs=-1 # Nutzt alle CPU-Kerne für schnelleres Training
)

rf.fit(X_train, y_train)

# 6. Modell testen und auswerten
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n>>> Modell Genauigkeit (Accuracy): {accuracy * 100:.2f}% <<<")

print("\nWichtigkeit der Eigenschaften (Feature Importances):")
for feature, importance in zip(features, rf.feature_importances_):
    print(f" - {feature}: {importance * 100:.1f}%")

# 7. Trainiertes Modell speichern
# Erstellt den Ordner 'data/processed', falls er noch nicht existiert
os.makedirs(os.path.dirname(model_output), exist_ok=True)
joblib.dump(rf, model_output)

print(f"\nErfolgreich! Modell gespeichert unter: {model_output}")