# FastAPI - TFM Emisiones Industriales UE
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

app = FastAPI(
    title="TFM - Emisiones Industriales UE",
    description="API para predicción de emisiones CO₂, clasificación de sector, clustering de países y proyección temporal",
    version="1.0.0"
)

# Carga de modelos
model_co2      = joblib.load('modelos/model_co2.pkl')
model_sector   = tf.keras.models.load_model('modelos/model_sector.keras')
model_clusters = joblib.load('modelos/model_clusters.pkl')
model_series   = joblib.load('modelos/model_series.pkl')
scaler         = joblib.load('modelos/scaler.pkl')

#  Features del modelo CO2
CO2_FEATURES = [
    'reportingYear', 'Latitude', 'Longitude',
    'WASTE_Recovery_HW', 'AIR_Nitrogen oxides (NOX)',
    'WASTE_Disposal_HW', 'WASTE_Recovery_NONHW', 'WASTE_Disposal_NONHW',
    'EPRTR_SectorName_Chemical industry',
    'EPRTR_SectorName_Energy sector',
    'EPRTR_SectorName_Mineral industry',
    'EPRTR_SectorName_Other activities',
    'EPRTR_SectorName_Paper and wood production and processing',
    'EPRTR_SectorName_Production and processing of metals',
    'EPRTR_SectorName_Waste and wastewater management',
    'countryName_Belgium', 'countryName_Bulgaria', 'countryName_Croatia',
    'countryName_Cyprus', 'countryName_Czechia', 'countryName_Denmark',
    'countryName_Estonia', 'countryName_Finland', 'countryName_France',
    'countryName_Germany', 'countryName_Greece', 'countryName_Hungary',
    'countryName_Ireland', 'countryName_Italy', 'countryName_Lithuania',
    'countryName_Luxembourg', 'countryName_Netherlands', 'countryName_Norway',
    'countryName_Other', 'countryName_Poland', 'countryName_Portugal',
    'countryName_Romania', 'countryName_Slovakia', 'countryName_Slovenia',
    'countryName_Spain', 'countryName_Sweden', 'countryName_Switzerland',
    'countryName_United Kingdom'
]

# Nombres de sectores
SECTOR_NAMES = {
    0: 'Sector energético',
    1: 'Producción y transformación de metales',
    2: 'Industria mineral',
    3: 'Industria química',
    4: 'Gestión de residuos y aguas residuales',
    5: 'Producción y transformación de papel y madera',
    6: 'Producción ganadera intensiva y acuicultura',
    7: 'Productos alimentarios y de bebidas',
    8: 'Otras actividades'
}

# Lista de features del modelo de clasificación (orden exacto del scaler)
SECTOR_FEATURES = [
    'AIR_Ammonia (NH3)', 'AIR_Carbon dioxide (CO2)',
    'AIR_Carbon dioxide (CO2) excluding biomass', 'AIR_Carbon monoxide (CO)',
    'AIR_Chlorine and inorganic compounds (as HCl)',
    'AIR_Hydrochlorofluorocarbons (HCFCs)', 'AIR_Mercury and compounds (as Hg)',
    'AIR_Methane (CH4)', 'AIR_Nickel and compounds (as Ni)',
    'AIR_Nitrogen oxides (NOX)', 'AIR_Nitrous oxide (N2O)',
    'AIR_Non-methane volatile organic compounds (NMVOC)',
    'AIR_Particulate matter (PM10)', 'AIR_Sulphur oxides (SOX)',
    'AIR_Zinc and compounds (as Zn)', 'WASTE_Disposal_HW',
    'WASTE_Disposal_NONHW', 'WASTE_Recovery_HW', 'WASTE_Recovery_NONHW',
    'WATER_Arsenic and compounds (as As)', 'WATER_Chlorides (as total Cl)',
    'WATER_Copper and compounds (as Cu)', 'WATER_Fluorides (as total F)',
    'WATER_Lead and compounds (as Pb)', 'WATER_Nickel and compounds (as Ni)',
    'WATER_Total nitrogen', 'WATER_Total organic carbon(as total C or COD/3) (TOC)',
    'WATER_Total phosphorus', 'WATER_Zinc and compounds (as Zn)',
    'TRANSFER_Nickel and compounds (as Ni)', 'TRANSFER_Phenols (as total C)',
    'TRANSFER_Total nitrogen', 'TRANSFER_Total organic carbon(as total C or COD/3) (TOC)',
    'TRANSFER_Total phosphorus', 'TRANSFER_Zinc and compounds (as Zn)',
    'reported_AIR_Ammonia (NH3)', 'reported_AIR_Carbon dioxide (CO2)',
    'reported_AIR_Carbon dioxide (CO2) excluding biomass',
    'reported_AIR_Carbon monoxide (CO)',
    'reported_AIR_Chlorine and inorganic compounds (as HCl)',
    'reported_AIR_Hydrochlorofluorocarbons (HCFCs)',
    'reported_AIR_Mercury and compounds (as Hg)', 'reported_AIR_Methane (CH4)',
    'reported_AIR_Nickel and compounds (as Ni)', 'reported_AIR_Nitrogen oxides (NOX)',
    'reported_AIR_Nitrous oxide (N2O)',
    'reported_AIR_Non-methane volatile organic compounds (NMVOC)',
    'reported_AIR_Particulate matter (PM10)', 'reported_AIR_Sulphur oxides (SOX)',
    'reported_AIR_Zinc and compounds (as Zn)', 'reported_WASTE_Disposal_HW',
    'reported_WASTE_Disposal_NONHW', 'reported_WASTE_Recovery_HW',
    'reported_WASTE_Recovery_NONHW', 'reported_WATER_Arsenic and compounds (as As)',
    'reported_WATER_Chlorides (as total Cl)', 'reported_WATER_Copper and compounds (as Cu)',
    'reported_WATER_Fluorides (as total F)', 'reported_WATER_Lead and compounds (as Pb)',
    'reported_WATER_Nickel and compounds (as Ni)', 'reported_WATER_Total nitrogen',
    'reported_WATER_Total organic carbon(as total C or COD/3) (TOC)',
    'reported_WATER_Total phosphorus', 'reported_WATER_Zinc and compounds (as Zn)',
    'reported_TRANSFER_Nickel and compounds (as Ni)',
    'reported_TRANSFER_Phenols (as total C)', 'reported_TRANSFER_Total nitrogen',
    'reported_TRANSFER_Total organic carbon(as total C or COD/3) (TOC)',
    'reported_TRANSFER_Total phosphorus', 'reported_TRANSFER_Zinc and compounds (as Zn)'
]

@app.post('/predict/sector')
def predict_sector(data: dict):
    # Crear DataFrame con el orden exacto del scaler
    df = pd.DataFrame([data])

    # Asegurar que tiene todas las features en el orden correcto
    for col in SECTOR_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[SECTOR_FEATURES]

    # Aplicar log1p igual que en el entrenamiento
    df = np.log1p(df)

    # Escalar
    df_scaled = scaler.transform(df)

    # Predecir
    prediccion = model_sector.predict(df_scaled, verbose=0)
    sector_idx = int(np.argmax(prediccion[0]))
    confianza = round(float(prediccion[0][sector_idx]) * 100, 1)

    return {
        'sector_predicho': SECTOR_NAMES.get(sector_idx, str(sector_idx)),
        'sector_codigo': sector_idx + 1,
        'confianza_pct': confianza
    }

CLUSTER_NAMES = {
    0: 'Atípicos — vertidos al agua muy elevados',
    1: 'Núcleo industrial europeo',
    2: 'Perfil industrial reducido'
}

# ── Endpoints

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/predict/co2')
def predict_co2(data: dict):
    df = pd.DataFrame([data])
    # Asegurar que tiene todas las features necesarias
    for col in CO2_FEATURES:
        if col not in df.columns:
            df[col] = 0
    df = df[CO2_FEATURES]

    # Debug — ver qué llega al modelo
    print("Features enviadas al modelo:")
    print(df.to_dict())

    prediccion_log = model_co2.predict(df)[0]
    print(f"Predicción log10: {prediccion_log}")

    toneladas = round(float(10 ** prediccion_log), 0)
    return {
        'co2_predicho_toneladas': toneladas,
        'co2_predicho_Mt': round(toneladas / 1_000_000, 3)
    }

@app.post('/predict/sector')
def predict_sector(data: dict):
    df = pd.DataFrame([data])
    # Asegurar 70 features
    for col in range(70 - len(df.columns)):
        df[f'feature_{col}'] = 0
    df_scaled = scaler.transform(df.iloc[:, :70])
    prediccion = model_sector.predict(df_scaled)
    sector_idx = int(np.argmax(prediccion[0]))
    confianza = round(float(prediccion[0][sector_idx]) * 100, 1)
    return {
        'sector_predicho': SECTOR_NAMES.get(sector_idx, str(sector_idx)),
        'sector_codigo': sector_idx + 1,
        'confianza_pct': confianza
    }

@app.post('/predict/cluster')
def predict_cluster(data: dict):
    df = pd.DataFrame([data])
    # KMeans espera 35 features
    for col in range(35 - len(df.columns)):
        df[f'feature_{col}'] = 0
    X = np.log1p(df.iloc[:, :35].values)
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    cluster = int(model_clusters.predict(X_scaled)[0])
    return {
        'cluster': cluster,
        'descripcion': CLUSTER_NAMES.get(cluster, str(cluster))
    }

@app.post('/predict/series')
def predict_series(data: dict):
    anio = int(data.get('anio', 2027))
    if anio < 2025 or anio > 2030:
        return {'error': 'El año debe estar entre 2025 y 2030'}

    serie = model_series['serie_entrenamiento']
    # Modelo Drift: prolonga la tendencia media
    slope = (serie.iloc[-1] - serie.iloc[0]) / (len(serie) - 1)
    horizon = anio - 2024
    prediccion_log = serie.iloc[-1] + slope * horizon
    toneladas = round(float(10 ** prediccion_log), 0)
    return {
        'anio': anio,
        'co2_predicho_toneladas': toneladas,
        'co2_predicho_Mt': round(toneladas / 1_000_000, 3)
    }
