# Datasets Catalog — مصادر البيانات

A comprehensive catalog of all datasets used or referenced by the system.

---

## Category 1: Weather & Meteorological Data — بيانات الطقس

### 1.1 ERA5 Reanalysis (ECMWF)
- **Description:** Global reanalysis dataset at 0.25° resolution (~28 km), hourly, from 1940–present
- **Variables:** Air temperature, humidity, wind speed/direction, solar radiation, pressure
- **Use:** Historical weather baseline and feature validation
- **Source:** https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
- **API:** https://cds.climate.copernicus.eu/how-to-api
- **Access:** Free (requires registration + API key)

### 1.2 Open-Meteo API
- **Description:** Free weather API sourcing from 80+ national weather services, with 16-day forecast
- **Variables:** Temperature, humidity, wind, radiation (direct + diffuse), UV, cloud cover
- **Use:** **Primary real-time weather source** (no API key needed)
- **Source:** https://open-meteo.com/
- **GitHub:** https://github.com/open-meteo/open-meteo
- **Access:** Free (no API key for non-commercial use)

### 1.3 Weather2K Dataset
- **Description:** 2,130 weather stations, 3 years (2017-2021), 20 variables
- **Variables:** Temperature, humidity, wind, pressure, visibility, etc.
- **Use:** Time-series model training and validation
- **Source:** https://github.com/bycnfz/weather2k
- **HuggingFace:** https://huggingface.co/datasets/BUPT-PRIS-727/Weather2K
- **Access:** Free

---

## Category 2: Heat & Thermal Comfort — الراحة الحرارية

### 2.1 GloUTCI-M (Global UTCI Dataset)
- **Description:** Global monthly UTCI at 1 km resolution (2000-2022)
- **Variables:** UTCI (Universal Thermal Climate Index)
- **Use:** **Primary ML target/label** for heat exposure prediction
- **Source:** https://doi.org/10.5281/zenodo.8310513
- **Access:** Free

### 2.2 WBGT Dataset (US Counties, 2000-2020)
- **Description:** County-level WBGT for all US counties
- **Variables:** WBGT, Heat Index, Humidex, UTCI, and other indices
- **Use:** Model validation and cross-comparison
- **Source:** https://doi.org/10.6084/m9.figshare.19419836
- **Paper:** https://www.nature.com/articles/s41597-022-01405-3
- **Access:** Free

### 2.3 WBGT Global (ERA5-based, 1979-2022)
- **Description:** Global WBGT derived from ERA5 reanalysis
- **Variables:** WBGT* = 0.7Tw + 0.3Td
- **Use:** Alternative WBGT label for global coverage
- **Source:** https://zenodo.org/records/10428575
- **Access:** Free

---

## Category 3: Land Surface Temperature (LST) — حرارة سطح الأرض

### 3.1 MODIS LST (NASA Earthdata)
- **Description:** Daily LST from Terra/Aqua at 1 km resolution
- **Products:** MOD11A1 (Terra), MYD11A1 (Aqua), MOD21A1D, MYD21A1D
- **Use:** Feature — actual road surface temperature estimation
- **Source:** https://www.earthdata.nasa.gov/data/instruments/modis
- **Access:** Free (requires Earthdata Login)

### 3.2 Landsat LST
- **Description:** Higher resolution LST from Landsat (30m)
- **Use:** Urban-scale surface temperature analysis
- **Source:** https://landsatlst.appspot.com/
- **Access:** Free

### 3.3 LandBench 1.0
- **Description:** Benchmark dataset for AI land surface prediction
- **Use:** Advanced LST model training
- **Source:** https://github.com/ecmwf-lab/ai-models
- **Access:** Free

---

## Category 4: Roads, Buildings & Urban Data — بيانات الطرق والمباني

### 4.1 OpenStreetMap (OSM)
- **Description:** Open collaborative map with roads, buildings, land use
- **Use:** **Primary road network and urban structure source**
- **Source:** https://www.openstreetmap.org/
- **API:** https://overpass-api.de/
- **Access:** Free

### 4.2 OSMnx Python Package
- **Description:** Python library for downloading and analyzing OSM data
- **Use:** Graph construction and spatial analysis
- **Source:** https://osmnx.readthedocs.io/
- **GitHub:** https://github.com/gboeing/osmnx
- **Access:** Free

### 4.3 Urban Heat Island (UHI) Monitoring Dataset
- **Description:** Urban heat island data across multiple cities
- **Use:** Urban-rural temperature differential analysis
- **Source:** https://www.kaggle.com/datasets/atharvasoundankar/urban-heat-island-uhi-monitoring-dataset
- **Access:** Free (requires Kaggle account)

---

## Category 5: Thermal Comfort Modeling — نمذجة الراحة الحرارية

### 5.1 SOLWEIG Model
- **Description:** Model for computing Mean Radiant Temperature and shadow patterns
- **Use:** Reference for shadow modeling validation
- **Source:** https://github.com/UMEP-dev/solweig
- **Docs:** https://umep-docs.readthedocs.io/en/latest/OtherManuals/SOLWEIG.html
- **Access:** Free

### 5.2 ML Heat Index Dataset (US Cities)
- **Description:** ERA5-derived features mapped to Heat Index for US cities
- **Use:** ML model training data
- **Source:** https://arxiv.org/html/2603.19488v1
- **Access:** Free

---

## Summary Table

| # | Dataset | Purpose | Source |
|---|---------|---------|--------|
| 1 | OpenStreetMap | Roads + buildings | OSM |
| 2 | Open-Meteo API | Real-time weather + forecast | open-meteo.com |
| 3 | ERA5 (CDS) | Historical weather (high quality) | CDS |
| 4 | MODIS LST (NASA) | Surface temperature | Earthdata |
| 5 | GloUTCI-M | Thermal comfort labels (ML target) | Zenodo |
| 6 | WBGT US Counties | Heat stress validation | Figshare |
| 7 | Weather2K | Time-series training | GitHub |
| 8 | UHI Kaggle | Urban heat island effects | Kaggle |

---

## Important Notes

1. **MODIS LST** at 1 km resolution may be too coarse for street-level analysis. Use as a **feature**, not as the primary **target**.
2. **GloUTCI-M** at 1 km monthly resolution can serve as the training **label** when combined with higher-frequency features.
3. Not every dataset is needed simultaneously — each fills a specific role in the pipeline.
4. The primary **target (label)** for ML training combines WBGT or Heat Index with duration of exposure.
5. **SOLWEIG** can generate **synthetic** shadow/radiation data for locations where real measurements are unavailable.
