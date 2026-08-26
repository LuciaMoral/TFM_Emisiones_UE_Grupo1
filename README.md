# TFM — Análisis de Emisiones Industriales en la Unión Europea
**Registro Europeo E-PRTR / IED · Agencia Europea de Medio Ambiente · 2007–2024**

---

## GRUPO 1 - Integrantes:

- Esneder Arenas
- Shiyi Chen
- William Ventura
- Brayan Carhuancho
- Lucía Moral 


---

## Contenido del repositorio

- README.md
- TFM_Emisiones_UE_FINAL.ipynb: notebook completo ejecutado
- TFM_Emisiones_UE_FINAL.html: exportado para visualización
- Carpeta API:
      - main.py (servidor FastAPI)
      - requirements.txt
      - Dockerfile: despliegue en Hugging Face Spaces.
- Carpeta con modelos en pkl.


  ---
  ## Para ver el análisis completo:

  1. Descarga `TFM_Emisiones_UE_FINAL.html`, y abrelo en cualquier navegador.
  2. API en producción:
     - Web: enlace pendiente.
    

## Ejecutar la API en local

# 1. Instalar dependencias
pip install -r api/requirements.txt

# 2. Arrancar el servidor
uvicorn api.main:app --reload

# 3. Abrir en el navegador
http://localhost:8000


---

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
