# System Architecture — التوثيق التفصيلي للبنية

## Overview

The Heat Exposure Prediction & Route Optimization System is composed of
six processing layers, each responsible for a specific stage of the pipeline.

---

## Layer 1: Data Ingestion — طبقة جمع البيانات

Responsible for fetching raw data from external APIs and local sources.

| Module | Description | Technology |
|--------|-------------|------------|
| `weather_client.py` | Hourly weather (temp, humidity, wind, radiation) | Open-Meteo API |
| `osm_fetcher.py` | Road networks, buildings, trees | OSMnx / Overpass |
| `lst_fetcher.py` | Land surface temperature estimation | MODIS (placeholder) |
| `elevation_fetcher.py` | Terrain elevation | Open-Meteo Elevation |

### Data Flow
```
Open-Meteo API  →  WeatherClient  →  WeatherSnapshot / WeatherForecast
OSM / Overpass  →  OSMFetcher     →  UrbanData (graph + buildings + trees)
NASA Earthdata  →  LSTFetcher     →  LSTEstimate
```

---

## Layer 2: Feature Engineering — هندسة الميزات

Transforms raw data into ML-ready features for each road segment.

| Module | Description |
|--------|-------------|
| `solar_calculator.py` | Sun altitude, azimuth (pvlib) |
| `shadow_estimator.py` | Shade fraction per segment (2.5D model) |
| `surface_classifier.py` | Surface type → thermal properties |
| `heat_index_calculator.py` | Heat Index, WBGT, Apparent Temperature |
| `segment_extractor.py` | Split edges into 50m segments |

### Shadow Model (2.5D)
```
shadow_length = building_height / tan(solar_altitude)
shadow_direction = solar_azimuth + 180°
shade_fraction = f(shadow_footprint ∩ road_segment)
```

### Feature Vector (per segment)
```
[air_temp, humidity, wind_speed, direct_radiation, diffuse_radiation,
 shade_fraction, surface_heat_factor, solar_altitude, hour_of_day, LST]
```

---

## Layer 3: ML Modeling — النمذجة

| Model | Algorithm | Purpose |
|-------|-----------|---------|
| Heat Exposure Model | XGBoost / RandomForest | Predict heat exposure score |
| Time Series Model | Sinusoidal / Prophet | Forecast daily temperature profile |

### Target Variable
The heat exposure score is a composite index inspired by WBGT and Heat Index,
representing the expected thermal stress on a person traversing the segment.

### Heuristic Model (when untrained)
```
score = (0.35 × temp_norm + 0.15 × humidity_norm + 0.20 × radiation_norm
         + 0.15 × surface_factor + 0.15 × (1 - shade)) × wind_factor × 55
```

---

## Layer 4: Optimization — التحسين

### Combined Cost Function
```
cost(edge) = α × travel_time + β × cumulative_heat_exposure
```

Where:
- `α` = time weight (0-1)
- `β` = heat weight (0-1)
- `cumulative_heat_exposure = heat_score × segment_duration`

### Presets
| Preset | α | β |
|--------|---|---|
| Fastest | 1.0 | 0.0 |
| Slightly Fast | 0.7 | 0.3 |
| Balanced | 0.5 | 0.5 |
| Slightly Cool | 0.3 | 0.7 |
| Coolest | 0.0 | 1.0 |

### Algorithms
- **Dijkstra's** for single-objective shortest path
- **K-shortest simple paths** for route alternatives
- **Greedy + permutation search** for multi-trip scheduling

---

## Layer 5: Explainability — الشرح والتفسير

| Module | Purpose |
|--------|---------|
| SHAP Explainer | Feature importance per prediction |
| Route Comparator | Side-by-side route metrics |
| Recommendation Engine | Natural-language suggestions |

---

## Layer 6: Presentation — واجهة المستخدم

- **FastAPI** REST API with OpenAPI docs
- **Streamlit** interactive dashboard with Plotly charts
- **Folium** map visualization (route coloring by heat)
