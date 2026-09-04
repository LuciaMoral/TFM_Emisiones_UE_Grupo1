# FastAPI - TFM Emisiones Industriales UE
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import joblib
import tensorflow as tf
import pandas as pd
import numpy as np



app = FastAPI(
    title="TFM - Emisiones Industriales UE",
    description="API para predicción de emisiones CO₂, clasificación de sector, clustering de países y proyección temporal",
    version="1.0.0"
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Carga de modelos & scalers
model_co2      = joblib.load('modelos/model_co2.pkl')
model_sector   = tf.keras.models.load_model('modelos/model_sector.keras')
model_clusters = joblib.load('modelos/model_clusters.pkl')
model_series   = joblib.load('modelos/model_series.pkl')
scaler         = joblib.load('modelos/scaler.pkl')
scaler_clusters = joblib.load('modelos/scaler_clusters.pkl')

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

# cluster names

CLUSTER_NAMES = {
    0: 'Atípicos — vertidos al agua muy elevados',
    1: 'Núcleo industrial europeo',
    2: 'Perfil industrial reducido'
}

POLLUTANT_COLS_CLUSTER = [
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
    'TRANSFER_Total phosphorus', 'TRANSFER_Zinc and compounds (as Zn)'
]
# ── Endpoints
@app.get('/', response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TFM - Emisiones Industriales UE</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>

<header>
    <h1>Análisis de Emisiones Industriales en la UE</h1>
    <p>Trabajo de Fin de Máster &nbsp;·&nbsp; Datos E-PRTR / IED · Agencia Europea de Medio Ambiente (2007–2024)</p>
</header>

<div class="container">

    <div class="tabs">
        <button class="tab active" onclick="showTab('co2', this)">Predicción CO₂</button>
        <button class="tab" onclick="showTab('sector', this)">Identificar Sector</button>
        <button class="tab" onclick="showTab('cluster', this)">Cluster País</button>
        <button class="tab" onclick="showTab('series', this)">Proyección 2025-2030</button>
    </div>

    <!-- PANEL CO2 -->
    <div id="panel-co2" class="panel active">
        <div class="card">
            <h2>Predicción de emisiones de CO₂</h2>
            <p class="desc">Introduce el perfil de una instalación industrial y el modelo estimará sus emisiones anuales de CO₂.</p>
            <div class="grid">
                <div class="field">
                    <label>Año de reporte (2007-2024)</label>
                    <input type="number" id="co2_year" value="2022" min="2007" max="2024">
                </div>
                <div class="field">
                    <label>Sector industrial</label>
                    <select id="co2_sector">
                        <option value="Energy sector">Sector energético</option>
                        <option value="Chemical industry">Industria química</option>
                        <option value="Mineral industry">Industria mineral</option>
                        <option value="Production and processing of metals">Metales</option>
                        <option value="Waste and wastewater management">Gestión de residuos</option>
                        <option value="Paper and wood production and processing">Papel y madera</option>
                        <option value="Intensive livestock production and aquaculture">Ganadería intensiva</option>
                        <option value="Animal and vegetable products from the food and beverage sector">Alimentación</option>
                    </select>
                </div>
                <div class="field">
                    <label>País</label>
                    <select id="co2_country">
                        <option value="Germany">Alemania</option>
                        <option value="Spain">España</option>
                        <option value="France">Francia</option>
                        <option value="Poland">Polonia</option>
                        <option value="Italy">Italia</option>
                        <option value="United Kingdom">Reino Unido</option>
                        <option value="Netherlands">Países Bajos</option>
                        <option value="Belgium">Bélgica</option>
                        <option value="Other">Otro</option>
                    </select>
                </div>
                <div class="field">
                    <label>NOₓ — Óxidos de nitrógeno (kg/año)</label>
                    <input type="number" id="co2_nox" value="50000">
                </div>
                <div class="field">
                    <label>Residuos peligrosos recuperados (t/año)</label>
                    <input type="number" id="co2_waste_rec_hw" value="500">
                </div>
                <div class="field">
                    <label>Residuos peligrosos eliminados (t/año)</label>
                    <input type="number" id="co2_waste_dis_hw" value="100">
                </div>
                <div class="field">
                    <label>Residuos no peligrosos recuperados (t/año)</label>
                    <input type="number" id="co2_waste_rec_nonhw" value="5000">
                </div>
                <div class="field">
                    <label>Residuos no peligrosos eliminados (t/año)</label>
                    <input type="number" id="co2_waste_dis_nonhw" value="200">
                </div>
            </div>
            <button class="btn" onclick="predecirCO2()">Calcular predicción</button>
            <div class="resultado" id="res-co2">
                <div class="label">CO₂ estimado</div>
                <div class="valor" id="res-co2-valor">—</div>
                <div class="sub" id="res-co2-sub"></div>
            </div>
            <p class="nota">Modelo XGBoost entrenado sobre 33.831 instalaciones europeas (2007–2024).</p>
        </div>
    </div>

    <!-- PANEL SECTOR -->
    <div id="panel-sector" class="panel">
        <div class="card">
            <h2>Identificación del sector industrial</h2>
            <p class="desc">Introduce el perfil de emisiones de una instalación y el modelo identificará a qué sector pertenece.</p>
            <div class="grid">
                <div class="field">
                    <label>Amoniaco NH₃ (kg/año)</label>
                    <input type="number" id="sec_nh3" value="50000">
                </div>
                <div class="field">
                    <label>CO₂ (kg/año)</label>
                    <input type="number" id="sec_co2" value="0">
                </div>
                <div class="field">
                    <label>NOₓ (kg/año)</label>
                    <input type="number" id="sec_nox" value="0">
                </div>
                <div class="field">
                    <label>Metano CH₄ (kg/año)</label>
                    <input type="number" id="sec_ch4" value="0">
                </div>
                <div class="field">
                    <label>Residuos no peligrosos recuperados (t/año)</label>
                    <input type="number" id="sec_waste_nonhw" value="5000">
                </div>
                <div class="field">
                    <label>SOₓ — Óxidos de azufre (kg/año)</label>
                    <input type="number" id="sec_sox" value="0">
                </div>
            </div>
            <button class="btn" onclick="predecirSector()">Identificar sector</button>
            <div class="resultado" id="res-sector">
                <div class="label">Sector identificado</div>
                <div class="valor" id="res-sector-valor">—</div>
                <div class="sub" id="res-sector-confianza"></div>
            </div>
            <p class="nota">Red neuronal entrenada sobre 81.005 instalaciones y 9 sectores industriales E-PRTR.</p>
        </div>
    </div>

    <!-- PANEL CLUSTER -->
    <div id="panel-cluster" class="panel">
        <div class="card">
            <h2>Cluster de país por perfil de emisiones</h2>
            <p class="desc">Introduce el perfil medio de emisiones por instalación de un país y el modelo identificará a qué grupo pertenece.</p>
            <div class="grid">
                <div class="field">
                    <label>CO₂ medio por instalación (kg/año)</label>
                    <input type="text" id="cl_co2" value="985.944.900"  oninput="formatearMiles(this)">
                </div>
                <div class="field">
                    <label>NOₓ medio por instalación (kg/año)</label>
                    <input type="text" id="cl_nox" value="1.971.553" oninput="formatearMiles(this)">
                </div>
                <div class="field">
                    <label>SOₓ medio por instalación (kg/año)</label>
                    <input type="text" id="cl_sox" value="13.604.870"  oninput="formatearMiles(this)">
                </div>
                <div class="field">
                    <label>NH₃ medio por instalación (kg/año)</label>
                    <input type="text" id="cl_nh3" value="14.663" oninput="formatearMiles(this)">
                </div>
                <div class="field">
                    <label>Cloruros en agua (kg/año)</label>
                    <input type="text" id="cl_chlorides" value="7.451.533"  oninput="formatearMiles(this)>
                </div>
                <div class="field">
                    <label>Residuos no peligrosos recuperados (t/año)</label>
                    <input type="text" id="cl_waste_nonhw" value="85.922"  oninput="formatearMiles(this)">
                </div>
            </div>
            <button class="btn" onclick="predecirCluster()">Identificar cluster</button>
            <div class="resultado" id="res-cluster">
                <div class="label">Grupo identificado</div>
                <div class="valor" id="res-cluster-valor">—</div>
                <div class="sub" id="res-cluster-desc"></div>
            </div>
            <p class="nota">KMeans aplicado sobre 32 países europeos con 35 variables de emisiones (2007–2024).</p>
        </div>
    </div>

    <!-- PANEL SERIES -->
    <div id="panel-series" class="panel">
        <div class="card">
            <h2>Proyección de emisiones CO₂ (2025-2030)</h2>
            <p class="desc">Selecciona un año entre 2025 y 2030 para obtener la proyección de emisiones industriales de CO₂ en Europa.</p>
            <div class="grid">
                <div class="field">
                    <label>Año de proyección</label>
                    <select id="ser_anio">
                        <option value="2025">2025</option>
                        <option value="2026">2026</option>
                        <option value="2027" selected>2027</option>
                        <option value="2028">2028</option>
                        <option value="2029">2029</option>
                        <option value="2030">2030</option>
                    </select>
                </div>
            </div>
            <button class="btn" onclick="predecirSeries()">Calcular proyección</button>
            <div class="resultado" id="res-series">
                <div class="label">CO₂ proyectado</div>
                <div class="valor" id="res-series-valor">—</div>
                <div class="sub" id="res-series-sub"></div>
            </div>
            <p class="nota">Modelo Drift entrenado sobre la serie histórica 2007–2024 del registro E-PRTR.</p>
        </div>
    </div>

</div>

<footer>
    TFM · Análisis de Emisiones Industriales en la UE · 2026 &nbsp;|&nbsp;
    <a href="/docs">Documentación técnica de la API</a>
</footer>

<script>
    function showTab(name, btn) {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.getElementById('panel-' + name).classList.add('active');
        btn.classList.add('active');
    }

function getVal(id) {
    let val = document.getElementById(id).value
        .replace(/[.]/g, '')
        .replace(/[,]/g, '.');
    return parseFloat(val) || 0;
}
function formatearMiles(input) {
    let val = input.value.replace(/[.]/g, '');
    if (!isNaN(val) && val !== '') {
        input.value = Number(val).toLocaleString('es-ES');
    }
}
    function mostrarResultado(divId, valorId, valor, subId, sub) {
        const div = document.getElementById(divId);
        div.classList.remove('error');
        div.style.display = 'block';
        document.getElementById(valorId).innerText = valor;
        if (subId) document.getElementById(subId).innerText = sub || '';
    }

    function mostrarError(divId, valorId) {
        const div = document.getElementById(divId);
        div.classList.add('error');
        div.style.display = 'block';
        document.getElementById(valorId).innerText = 'Error al conectar con el modelo';
    }

    async function predecirCO2() {
        const sector = document.getElementById('co2_sector').value;
        const country = document.getElementById('co2_country').value;

        const datos = {
            'reportingYear': getVal('co2_year'),
            'Latitude': 51.5,
            'Longitude': 10.0,
            'WASTE_Recovery_HW': getVal('co2_waste_rec_hw'),
            'AIR_Nitrogen oxides (NOX)': getVal('co2_nox'),
            'WASTE_Disposal_HW': getVal('co2_waste_dis_hw'),
            'WASTE_Recovery_NONHW': getVal('co2_waste_rec_nonhw'),
            'WASTE_Disposal_NONHW': getVal('co2_waste_dis_nonhw'),
            [`EPRTR_SectorName_${sector}`]: 1,
            [`countryName_${country}`]: 1
        };

        try {
            const res = await fetch('/predict/co2', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(datos)
            });
            const data = await res.json();
            mostrarResultado(
                'res-co2', 'res-co2-valor',
                data.co2_predicho_toneladas.toLocaleString('es-ES') + ' t',
                'res-co2-sub',
                data.co2_predicho_Mt.toLocaleString('es-ES') + ' Mt'
            );
        } catch(e) {
            mostrarError('res-co2', 'res-co2-valor');
        }
    }

    async function predecirSector() {
        const datos = {
            'AIR_Ammonia (NH3)': getVal('sec_nh3'),
            'AIR_Carbon dioxide (CO2)': getVal('sec_co2'),
            'AIR_Nitrogen oxides (NOX)': getVal('sec_nox'),
            'AIR_Methane (CH4)': getVal('sec_ch4'),
            'WASTE_Recovery_NONHW': getVal('sec_waste_nonhw'),
            'AIR_Sulphur oxides (SOX)': getVal('sec_sox'),
            'reported_AIR_Ammonia (NH3)': getVal('sec_nh3') > 0 ? 1 : 0,
            'reported_AIR_Carbon dioxide (CO2)': getVal('sec_co2') > 0 ? 1 : 0,
            'reported_AIR_Nitrogen oxides (NOX)': getVal('sec_nox') > 0 ? 1 : 0,
            'reported_AIR_Methane (CH4)': getVal('sec_ch4') > 0 ? 1 : 0,
            'reported_WASTE_Recovery_NONHW': getVal('sec_waste_nonhw') > 0 ? 1 : 0,
            'reported_AIR_Sulphur oxides (SOX)': getVal('sec_sox') > 0 ? 1 : 0
        };

        try {
            const res = await fetch('/predict/sector', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(datos)
            });
            const data = await res.json();
            mostrarResultado(
                'res-sector', 'res-sector-valor',
                data.sector_predicho,
                'res-sector-confianza',
                'Confianza: ' + data.confianza_pct + '%'
            );
        } catch(e) {
            mostrarError('res-sector', 'res-sector-valor');
        }
    }

    async function predecirCluster() {
        const datos = {
            'AIR_Carbon dioxide (CO2)': getVal('cl_co2'),
            'AIR_Nitrogen oxides (NOX)': getVal('cl_nox'),
            'AIR_Sulphur oxides (SOX)': getVal('cl_sox'),
            'AIR_Ammonia (NH3)': getVal('cl_nh3'),
            'WATER_Chlorides (as total Cl)': getVal('cl_chlorides'),
            'WASTE_Recovery_NONHW': getVal('cl_waste_nonhw')
        };

        try {
            const res = await fetch('/predict/cluster', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(datos)
            });
            const data = await res.json();
            mostrarResultado(
                'res-cluster', 'res-cluster-valor',
                'Cluster ' + data.cluster,
                'res-cluster-desc',
                data.descripcion
            );
        } catch(e) {
            mostrarError('res-cluster', 'res-cluster-valor');
        }
    }

    async function predecirSeries() {
        const anio = parseInt(document.getElementById('ser_anio').value);

        try {
            const res = await fetch('/predict/series', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({'anio': anio})
            });
            const data = await res.json();
            mostrarResultado(
                'res-series', 'res-series-valor',
                data.co2_predicho_toneladas.toLocaleString('es-ES') + ' t',
                'res-series-sub',
                data.co2_predicho_Mt.toLocaleString('es-ES') + ' Mt'
            );
        } catch(e) {
            mostrarError('res-series', 'res-series-valor');
        }
    }
</script>

</body>
</html>
"""

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
    # revierte el log10 para obtener toneladas
    toneladas = round(float(10 ** prediccion_log), 0)
    return {
        'co2_predicho_toneladas': toneladas,
        'co2_predicho_Mt': round(toneladas / 1_000_000, 3)
    }

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
    # calcular confianza como porcentaje de la probabilidad del sector predicho
    confianza = round(float(prediccion[0][sector_idx]) * 100, 1)

    return {
        'sector_predicho': SECTOR_NAMES.get(sector_idx, str(sector_idx)),
        'sector_codigo': sector_idx + 1,
        'confianza_pct': confianza
    }
@app.post('/predict/cluster')
def predict_cluster(data: dict):
    df = pd.DataFrame([data])
    for col in POLLUTANT_COLS_CLUSTER:
        if col not in df.columns:
            df[col] = 0
    df = df[POLLUTANT_COLS_CLUSTER]
    X_log = np.log1p(df)
    X_scaled = scaler_clusters.transform(X_log)
    cluster = int(model_clusters.predict(X_scaled)[0])
    return {
        'cluster': cluster,
        'descripcion': CLUSTER_NAMES.get(cluster, str(cluster))
    }

@app.post('/predict/series')
def predict_series(data: dict):
    # valor por defecto 2027 si no se proporciona el año para que no de error la api
    anio = int(data.get('anio', 2027))
    if anio < 2025 or anio > 2030:
        return {'error': 'El año debe estar entre 2025 y 2030'}

    serie = model_series['serie_entrenamiento']
    # Modelo Drift: prolonga la tendencia media, calculando la pendiente de la serie y proyectándola hacia el año deseado
    slope = (serie.iloc[-1] - serie.iloc[0]) / (len(serie) - 1)
    horizon = anio - 2024
    prediccion_log = serie.iloc[-1] + slope * horizon
    #revierte el log10 para obtener toneladas
    toneladas = round(float(10 ** prediccion_log), 0)
    return {
        'anio': anio,
        'co2_predicho_toneladas': toneladas,
        'co2_predicho_Mt': round(toneladas / 1_000_000, 3)
    }
