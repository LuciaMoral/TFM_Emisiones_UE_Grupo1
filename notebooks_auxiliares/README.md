# Notebooks auxiliares

Esta carpeta contiene notebooks de apoyo para tareas administrativas del TFM.

---

## `exportar_html.ipynb`

Convierte el notebook master `TFM_Emisiones_UE_FINAL.ipynb` a formato HTML 
para su visualización en el navegador sin necesidad de ejecutar Jupyter.

**Cómo usarlo:**
1. Abre el notebook en Google Colab
2. Ejecuta la celda — aparecerá un botón para seleccionar el fichero `.ipynb`
3. Selecciona `TFM_Emisiones_UE_FINAL.ipynb` desde tu ordenador.
4. El HTML se descargará automáticamente a tu carpeta `Downloads`

---

## `verificar_modelos.ipynb`

Carga los modelos guardados en Drive y verifica:

- Número y nombres de features de cada modelo
- Compatibilidad de versiones
- Genera y guarda `scaler_clusters.pkl` para el clustering

**Modelos verificados:**
- `model_co2.pkl`:  XGBoost regresión (43 features)
- `model_sector.keras`: Red neuronal clasificación (70 features)
- `model_clusters.pkl`: KMeans clustering (35 features)
- `model_series.pkl`:  Drift series temporales
- `scaler.pkl`: StandardScaler clasificación
- `scaler_clusters.pkl`: StandardScaler clustering
