# TFM — Análisis de Emisiones Industriales en la Unión Europea
**Registro Europeo E-PRTR / IED · Agencia Europea de Medio Ambiente · 2007–2024**

---

## GRUPO 1 - Integrantes:

- Brayam Anthony Carhuancho Trucios
- Esneder Arenas González
- Lucía Moral Baena
- Shiyi Chen 
- William Ventura

## Fuente de datos

- **Registro:** European Pollutant Release and Transfer Register (E-PRTR)
- **Institución:** Agencia Europea de Medio Ambiente (EEA)
- **Periodo:** 2007–2024 | **Países:** 32 | **Instalaciones:** 10.788
- **Reglamento:** (CE) N°166/2006 del Parlamento Europeo

---

## Preguntas de investigación

1. ¿Es posible predecir las emisiones de CO₂ de una instalación industrial a partir de su perfil de contaminantes?
2. ¿Puede identificarse el sector industrial de una instalación únicamente a partir de lo que emite?
3. ¿Qué patrones comunes de comportamiento ambiental comparten los países europeos?
4. ¿Mantiene Europa una trayectoria de reducción compatible con los objetivos del Pacto Verde para 2030?

---

## Contenido del repositorio

📁 TFM_Emisiones_UE_Grupo1/
      README.md
      TFM_Emisiones_UE_FINAL.ipynb - notebook completo ejecutado
      TFM_Emisiones_UE_FINAL.html - exportado para visualización
      Informe_TFM_Grupo1.pdf - informe final. 
📁 api/
      main.py -  servidor FastAPI
      requirements.txt - dependencias
      Dockerfile  despliegue en Render
📁 modelos/
      model_co2.pkl - XGBoost regresión CO₂
      model_sector.keras - Red neuronal clasificación sector
      model_clusters.pkl - KMeans clustering países
      model_series.pkl - Drift series temporales
      scaler.pkl - StandardScaler clasificación
      scaler_clusters.pkl - StandardScaler clustering
📁 notebooks_auxiliares/
      exportar_html.ipynb - conversión notebook a HTML
      verificar_modelos.ipynb - verificación features y modelos
      README.md

## Datos

Los datos orginales provienen del registro E-PRTR de la 
Agencia Europea de Medio Ambiente y están disponibles para descarga en:

🔗 https://sdi.eea.europa.eu/data/3461f4ab-a3ee-4af2-bc11-95e651a8d0ba?path=%2FUser-friendly-CSV

**Ficheros utilizados:**
- `F1_4_Air_Releases_Facilities.csv` — Emisiones al aire (370.301 filas)
- `F2_4_Water_Releases_Facilities.csv` — Emisiones al agua (252.313 filas)
- `F3_2_Transfers_Facilities.csv` — Transferencias (65.475 filas)
- `F4_2_WasteTransfers_Facilities.csv` — Residuos (843.877 filas)

El dataset limpio e integrado se genera ejecutando la sección de 
preprocesamiento del notebook master — resultado: **81.005 filas × 45 columnas**.
  ---
## API en producción

🌐 **Formulario web:** https://tfm-emisiones.onrender.com  
📖 **Documentación técnica:** https://tfm-emisiones.onrender.com/docs

**Nota:** el servicio gratuito puede tardar aprox. 30 segundos en responder
si ha estado inactivo. La primera llamada de "health" lo reinicia.

### Opción A — Formulario web (recomendado como demo)

Accede a https://tfm-emisiones.onrender.com

El formulario incluye 4 pestañas — una por modelo — con valores 
precargados listos para usar. Las variables mostradas son las más 
relevantes según el análisis de cada modelo:

| Modelo | Criterio de selección de variables |
|--------|-----------------------------------|
| Predicción CO₂ | Variables más importantes según análisis SHAP |
| Clasificación sector | Variables más discriminantes según análisis SHAP |
| Clustering países | Variables con mayor peso en PC1 y PC2 del PCA |
| Proyección 2025-2030 | Solo el año — el modelo Drift no usa otras variables |

Las variables no introducidas se imputan a 0 (no reportadas).
Para una predicción con el perfil completo, usar la Opción B.

### Opción B — Documentación técnica completa

Accede a https://tfm-emisiones.onrender.com/docs

Permite introducir todas las variables de cada modelo en formato JSON.

**Ejemplos de predicción: (para usar en esta versión)**

#### `/predict/series` — Proyección CO₂ 2030
```json
{"anio": 2030}
```
Resultado esperado: ~593 millones de toneladas

#### `/predict/sector` — Instalación energética (CO₂ y NOX altos)
```json
{
    "AIR_Ammonia (NH3)": 0,
    "AIR_Carbon dioxide (CO2)": 2000000000,
    "AIR_Carbon dioxide (CO2) excluding biomass": 0,
    "AIR_Carbon monoxide (CO)": 500000,
    "AIR_Chlorine and inorganic compounds (as HCl)": 0,
    "AIR_Hydrochlorofluorocarbons (HCFCs)": 0,
    "AIR_Mercury and compounds (as Hg)": 0,
    "AIR_Methane (CH4)": 0,
    "AIR_Nickel and compounds (as Ni)": 0,
    "AIR_Nitrogen oxides (NOX)": 5000000,
    "AIR_Nitrous oxide (N2O)": 0,
    "AIR_Non-methane volatile organic compounds (NMVOC)": 0,
    "AIR_Particulate matter (PM10)": 0,
    "AIR_Sulphur oxides (SOX)": 3000000,
    "AIR_Zinc and compounds (as Zn)": 0,
    "WASTE_Disposal_HW": 0,
    "WASTE_Disposal_NONHW": 0,
    "WASTE_Recovery_HW": 0,
    "WASTE_Recovery_NONHW": 0,
    "WATER_Arsenic and compounds (as As)": 0,
    "WATER_Chlorides (as total Cl)": 0,
    "WATER_Copper and compounds (as Cu)": 0,
    "WATER_Fluorides (as total F)": 0,
    "WATER_Lead and compounds (as Pb)": 0,
    "WATER_Nickel and compounds (as Ni)": 0,
    "WATER_Total nitrogen": 0,
    "WATER_Total organic carbon(as total C or COD/3) (TOC)": 0,
    "WATER_Total phosphorus": 0,
    "WATER_Zinc and compounds (as Zn)": 0,
    "TRANSFER_Nickel and compounds (as Ni)": 0,
    "TRANSFER_Phenols (as total C)": 0,
    "TRANSFER_Total nitrogen": 0,
    "TRANSFER_Total organic carbon(as total C or COD/3) (TOC)": 0,
    "TRANSFER_Total phosphorus": 0,
    "TRANSFER_Zinc and compounds (as Zn)": 0,
    "reported_AIR_Ammonia (NH3)": 0,
    "reported_AIR_Carbon dioxide (CO2)": 1,
    "reported_AIR_Carbon dioxide (CO2) excluding biomass": 0,
    "reported_AIR_Carbon monoxide (CO)": 1,
    "reported_AIR_Chlorine and inorganic compounds (as HCl)": 0,
    "reported_AIR_Hydrochlorofluorocarbons (HCFCs)": 0,
    "reported_AIR_Mercury and compounds (as Hg)": 0,
    "reported_AIR_Methane (CH4)": 0,
    "reported_AIR_Nickel and compounds (as Ni)": 0,
    "reported_AIR_Nitrogen oxides (NOX)": 1,
    "reported_AIR_Nitrous oxide (N2O)": 0,
    "reported_AIR_Non-methane volatile organic compounds (NMVOC)": 0,
    "reported_AIR_Particulate matter (PM10)": 0,
    "reported_AIR_Sulphur oxides (SOX)": 1,
    "reported_AIR_Zinc and compounds (as Zn)": 0,
    "reported_WASTE_Disposal_HW": 0,
    "reported_WASTE_Disposal_NONHW": 0,
    "reported_WASTE_Recovery_HW": 0,
    "reported_WASTE_Recovery_NONHW": 0,
    "reported_WATER_Arsenic and compounds (as As)": 0,
    "reported_WATER_Chlorides (as total Cl)": 0,
    "reported_WATER_Copper and compounds (as Cu)": 0,
    "reported_WATER_Fluorides (as total F)": 0,
    "reported_WATER_Lead and compounds (as Pb)": 0,
    "reported_WATER_Nickel and compounds (as Ni)": 0,
    "reported_WATER_Total nitrogen": 0,
    "reported_WATER_Total organic carbon(as total C or COD/3) (TOC)": 0,
    "reported_WATER_Total phosphorus": 0,
    "reported_WATER_Zinc and compounds (as Zn)": 0,
    "reported_TRANSFER_Nickel and compounds (as Ni)": 0,
    "reported_TRANSFER_Phenols (as total C)": 0,
    "reported_TRANSFER_Total nitrogen": 0,
    "reported_TRANSFER_Total organic carbon(as total C or COD/3) (TOC)": 0,
    "reported_TRANSFER_Total phosphorus": 0,
    "reported_TRANSFER_Zinc and compounds (as Zn)": 0
}
```
Resultado esperado: Sector energético

---

#### `/predict/cluster` — Perfil de España
```json
{
    "AIR_Ammonia (NH3)": 25000,
    "AIR_Carbon dioxide (CO2)": 450000000,
    "AIR_Carbon dioxide (CO2) excluding biomass": 30000000,
    "AIR_Carbon monoxide (CO)": 250000,
    "AIR_Chlorine and inorganic compounds (as HCl)": 800000,
    "AIR_Hydrochlorofluorocarbons (HCFCs)": 50,
    "AIR_Mercury and compounds (as Hg)": 8,
    "AIR_Methane (CH4)": 300000,
    "AIR_Nickel and compounds (as Ni)": 200,
    "AIR_Nitrogen oxides (NOX)": 600000,
    "AIR_Nitrous oxide (N2O)": 25000,
    "AIR_Non-methane volatile organic compounds (NMVOC)": 150000,
    "AIR_Particulate matter (PM10)": 80000,
    "AIR_Sulphur oxides (SOX)": 1500000,
    "AIR_Zinc and compounds (as Zn)": 300,
    "WASTE_Disposal_HW": 5000,
    "WASTE_Disposal_NONHW": 40000,
    "WASTE_Recovery_HW": 3000,
    "WASTE_Recovery_NONHW": 25000,
    "WATER_Arsenic and compounds (as As)": 50,
    "WATER_Chlorides (as total Cl)": 500000,
    "WATER_Copper and compounds (as Cu)": 80,
    "WATER_Fluorides (as total F)": 3000,
    "WATER_Lead and compounds (as Pb)": 30,
    "WATER_Nickel and compounds (as Ni)": 25,
    "WATER_Total nitrogen": 30000,
    "WATER_Total organic carbon(as total C or COD/3) (TOC)": 150000,
    "WATER_Total phosphorus": 2000,
    "WATER_Zinc and compounds (as Zn)": 400,
    "TRANSFER_Nickel and compounds (as Ni)": 50,
    "TRANSFER_Phenols (as total C)": 200,
    "TRANSFER_Total nitrogen": 3000,
    "TRANSFER_Total organic carbon(as total C or COD/3) (TOC)": 80000,
    "TRANSFER_Total phosphorus": 1500,
    "TRANSFER_Zinc and compounds (as Zn)": 500
}
```
Resultado esperado: Cluster 1 — Núcleo industrial europeo

---
#### `/predict/co2` — Instalación química española
```json
{
    "reportingYear": 2020,
    "Latitude": 40.4,
    "Longitude": -3.7,
    "WASTE_Recovery_HW": 200,
    "AIR_Nitrogen oxides (NOX)": 30000,
    "WASTE_Disposal_HW": 50,
    "WASTE_Recovery_NONHW": 3000,
    "WASTE_Disposal_NONHW": 100,
    "EPRTR_SectorName_Chemical industry": 1,
    "countryName_Spain": 1
}
```
Resultado esperado: predicción de toneladas de CO₂ para esa instalación

---

## Ejecutar la API en local

```bash
# 1. Crear entorno virtual con Python 3.11
python3.11 -m venv tfm
source tfm/bin/activate

# 2. Instalar dependencias
pip install -r api/requirements.txt

# 3. Arrancar el servidor
cd api
uvicorn main:app --reload

# 4. Abrir en el navegador
http://localhost:8000
```
> **Nota:** el modelo de CO₂ puede dar predicciones incorrectas en local
> por incompatibilidad de versiones de XGBoost. Usar la API en producción 
para predicciones fiables.


