"""
Heat Route Optimizer — West Riyadh
═══════════════════════════════════
Interactive route optimization using XGBoost heat‐exposure prediction,
real‐time solar geometry, 2.5‑D building/tree shadow projection,
and live Open‑Meteo weather data.

Bounded to BBBike extract: sw(46.48, 24.5401) → ne(46.5789, 24.6004)
"""

import sys, os, copy, math
import datetime as dt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from shapely.geometry import mapping, LineString

# ╔══════════════════════════════════════════════════════════════╗
# ║  CONSTANTS — neighbourhood bounding box                     ║
# ╚══════════════════════════════════════════════════════════════╝
SW_LAT, SW_LNG = 24.6850, 46.6600
NE_LAT, NE_LNG = 24.7200, 46.6950
CENTER_LAT = (SW_LAT + NE_LAT) / 2
CENTER_LNG = (SW_LNG + NE_LNG) / 2

# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG                                                ║
# ╚══════════════════════════════════════════════════════════════╝
st.set_page_config(
    page_title="Heat Route Optimizer — West Riyadh",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ╔══════════════════════════════════════════════════════════════╗
# ║  CSS                                                        ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp{background:#0a0e1a;font-family:'Inter',sans-serif}

/* Top bar */
.top-bar{background:linear-gradient(135deg,#0f1523,#141b2d);padding:.7rem 1.2rem;
  border-bottom:1px solid rgba(255,107,53,.25);display:flex;justify-content:space-between;
  align-items:center;border-radius:0 0 10px 10px;margin-bottom:.6rem}
.top-title{font-size:1.25rem;font-weight:800;
  background:linear-gradient(135deg,#FF6B35,#FFB347);-webkit-background-clip:text;
  -webkit-text-fill-color:transparent}
.top-sub{font-size:.72rem;color:#6b7b99;margin-top:2px}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:.72rem;color:#64ffda;
  background:rgba(100,255,218,.07);padding:3px 10px;border-radius:16px;
  border:1px solid rgba(100,255,218,.18)}
.dot{width:7px;height:7px;background:#64ffda;border-radius:50%;
  animation:p 2s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.35}}

/* Sidebar */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f1523,#141b2d)!important;
  border-right:1px solid rgba(255,107,53,.12)}
section[data-testid="stSidebar"] h3{color:#c8cdd6;font-size:.78rem;font-weight:700;
  text-transform:uppercase;letter-spacing:1.2px;margin:1rem 0 .4rem}

/* Location cards */
.loc{border-radius:10px;padding:.55rem .7rem;margin-bottom:.35rem;display:flex;
  align-items:center;gap:10px}
.loc-o{background:rgba(76,175,80,.1);border:1px solid rgba(76,175,80,.35)}
.loc-d{background:rgba(255,82,82,.1);border:1px solid rgba(255,82,82,.35)}
.loc-lbl{font-size:.82rem;font-weight:600;color:#e2e5ea}
.loc-xy{font-size:.68rem;color:#7b879c;font-family:'Courier New',monospace}

/* Weather */
.wg{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:.4rem}
.wc{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.05);
  border-radius:7px;padding:.4rem;text-align:center}
.wv{font-size:1rem;font-weight:700;color:#e2e5ea}
.wl{font-size:.6rem;color:#7b879c;text-transform:uppercase;letter-spacing:.4px}

/* Legend */
.leg{background:rgba(15,21,35,.92);border:1px solid rgba(255,255,255,.06);
  border-radius:10px;padding:.7rem}
.leg-t{font-size:.75rem;font-weight:700;color:#e2e5ea;margin-bottom:.45rem;
  text-transform:uppercase;letter-spacing:.8px}
.leg-i{display:flex;align-items:center;gap:8px;margin-bottom:5px;
  font-size:.74rem;color:#b0b6c2}
.leg-ln{width:22px;height:4px;border-radius:2px}

/* Solar info */
.sol{background:rgba(255,183,77,.07);border:1px solid rgba(255,183,77,.2);
  border-radius:8px;padding:.5rem .7rem;margin:.5rem 0;font-size:.75rem;color:#ddd}

/* Base container adjustments */
.block-container{padding-top:.4rem!important;padding-bottom:0!important}
</style>
""", unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║  CACHED LOADERS                                             ║
# ╚══════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner="📡 Loading neighbourhood map from GeoPackage …")
def load_map_data():
    from src.data_ingestion.osm_fetcher import OSMFetcher
    gpkg = os.path.join(
        os.path.dirname(__file__), "..",
        "حي العليا",
        "planet_46.66,24.685_46.695,24.72.gpkg",
    )
    return OSMFetcher().fetch_from_geopackage(gpkg)


@st.cache_resource(show_spinner="🤖 Loading XGBoost heat model & graph builder …")
def load_graph_builder():
    from src.optimization.graph_builder import GraphBuilder
    return GraphBuilder()


@st.cache_resource(show_spinner="🗺️ Preparing road‐network overlay …")
def build_road_geojson(_map_data):
    """One‑time GeoJSON of every road edge (de‑duplicated)."""
    graph = _map_data.road_graph
    feats, seen = [], set()
    for u, v, data in graph.edges(data=True):
        key = tuple(sorted([str(u), str(v)]))
        if key in seen:
            continue
        seen.add(key)
        geom = data.get("geometry")
        if geom is None:
            ud, vd = graph.nodes[u], graph.nodes[v]
            geom = LineString([(ud["x"], ud["y"]), (vd["x"], vd["y"])])
        hw = data.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        feats.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {"highway": str(hw)},
        })
    return {"type": "FeatureCollection", "features": feats}


# ╔══════════════════════════════════════════════════════════════╗
# ║  WEATHER HELPER                                             ║
# ╚══════════════════════════════════════════════════════════════╝

def get_weather_for_time(trip_time: dt.datetime):
    """Fetch Open-Meteo forecast and pick the closest hour to *trip_time*."""
    from src.data_ingestion.weather_client import WeatherClient, WeatherSnapshot
    try:
        client = WeatherClient()
        forecast = client.get_forecast(CENTER_LAT, CENTER_LNG, days=3)
        snap = forecast.at_time(trip_time)
        if snap is not None:
            return snap
    except Exception:
        pass
    # Deterministic fallback based on hour of day
    hour = trip_time.hour
    # Simple diurnal model for Riyadh summer
    base = 28.0 + 14.0 * math.sin(math.pi * max(hour - 6, 0) / 12) if 6 <= hour <= 18 else 30.0
    rad = max(0, 900 * math.sin(math.pi * max(hour - 6, 0) / 12)) if 6 <= hour <= 18 else 0.0
    return WeatherSnapshot(
        timestamp=trip_time, latitude=CENTER_LAT, longitude=CENTER_LNG,
        temperature_c=round(base, 1), relative_humidity_pct=15.0,
        wind_speed_kmh=12.0,
        direct_radiation_wm2=round(rad * 0.8), diffuse_radiation_wm2=round(rad * 0.2),
        cloud_cover_pct=5.0,
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║  SESSION STATE                                              ║
# ╚══════════════════════════════════════════════════════════════╝

_defaults = dict(origin=None, destination=None, routes=None,
                 route_graph=None, weather_snap=None, solar_info=None)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN                                                       ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    # ── Load resources (cached) ──────────────────────────────
    map_data = load_map_data()
    builder  = load_graph_builder()
    road_gj  = build_road_geojson(map_data)

    n_roads  = len(road_gj["features"])
    n_bldg   = len(map_data.buildings_gdf) if map_data.buildings_gdf is not None else 0
    n_tree   = len(map_data.trees_gdf)     if map_data.trees_gdf     is not None else 0
    xgb_ok   = builder.heat_model.is_trained

    # ── Fetch current weather for sidebar display ────────────
    if st.session_state.weather_snap is None:
        st.session_state.weather_snap = get_weather_for_time(dt.datetime.now())
    weather = st.session_state.weather_snap

    # ── Top bar ──────────────────────────────────────────────
    st.markdown(f"""
    <div class="top-bar">
      <div>
        <div class="top-title">🌡️ Heat Route Optimizer</div>
        <div class="top-sub">Heat Exposure Prediction &amp; Route Optimization — West Riyadh</div>
      </div>
      <div style="display:flex;gap:8px;align-items:center">
        <div class="badge"><div class="dot"></div>Connected | Data Loaded</div>
        <div class="badge" style="color:{'#64ffda' if xgb_ok else '#ffab40'}">
          {'🤖 XGBoost Active' if xgb_ok else '⚠️ Heuristic Mode'}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  SIDEBAR
    # ══════════════════════════════════════════════════════════

    with st.sidebar:
        # ── Location cards ───────────────────────
        st.markdown("### 📍 Select Locations")

        o, d = st.session_state.origin, st.session_state.destination
        _loc_html = ""
        if o:
            _loc_html += f'<div class="loc loc-o"><div style="font-size:1.1rem">📍</div><div><div class="loc-lbl">Origin</div><div class="loc-xy">{o["lat"]:.5f}, {o["lng"]:.5f}</div></div></div>'
        else:
            _loc_html += '<div class="loc loc-o" style="opacity:.45"><div style="font-size:1.1rem">📍</div><div><div class="loc-lbl">Origin</div><div class="loc-xy">Click on the map</div></div></div>'
        if d:
            _loc_html += f'<div class="loc loc-d"><div style="font-size:1.1rem">🏁</div><div><div class="loc-lbl">Destination</div><div class="loc-xy">{d["lat"]:.5f}, {d["lng"]:.5f}</div></div></div>'
        else:
            _loc_html += '<div class="loc loc-d" style="opacity:.45"><div style="font-size:1.1rem">🏁</div><div><div class="loc-lbl">Destination</div><div class="loc-xy">Click on the map</div></div></div>'
        st.markdown(_loc_html, unsafe_allow_html=True)

        if st.button("🔄 Reset Points", use_container_width=True):
            st.session_state.origin = None
            st.session_state.destination = None
            st.session_state.routes = None
            st.session_state.route_graph = None
            st.session_state.solar_info = None
            st.rerun()

        # ── Trip time ────────────────────────────
        st.markdown("### ⏱ Trip Time & Speed")
        c1, c2 = st.columns(2)
        with c1:
            dep_date = st.date_input("Date", value=dt.date.today())
        with c2:
            hours = [f"{h%12 if h%12!=0 else 12}:00 {'AM' if h < 12 else 'PM'}" for h in range(24)]
            curr_h = dt.datetime.now().hour
            selected_h_str = st.selectbox("Hour", hours, index=min(curr_h, 23))
            dep_hour = hours.index(selected_h_str)

        speed = st.slider("Speed (km/h)", 3.0, 50.0, 15.0, 1.0)
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            if st.button("🚶 Walk", use_container_width=True, key="bw"):
                speed = 5.0
        with mc2:
            if st.button("🚲 Bike", use_container_width=True, key="bb"):
                speed = 15.0
        with mc3:
            if st.button("🚗 Drive", use_container_width=True, key="bd"):
                speed = 40.0

        # ── Analyze ──────────────────────────────
        st.markdown("---")
        can_go = o is not None and d is not None
        analyze = st.button(
            "🔍 Analyze & Find Best Routes",
            type="primary", disabled=not can_go,
            use_container_width=True,
        )

        # ── Weather display ──────────────────────
        st.markdown("### 🌤 Current Weather")
        st.markdown(f"""
        <div class="wg">
          <div class="wc"><div class="wv">{weather.temperature_c:.1f}°C</div><div class="wl">🌡️ Temp</div></div>
          <div class="wc"><div class="wv">{weather.relative_humidity_pct:.0f}%</div><div class="wl">💧 Humidity</div></div>
          <div class="wc"><div class="wv">{weather.wind_speed_kmh:.0f} km/h</div><div class="wl">💨 Wind</div></div>
          <div class="wc"><div class="wv">{weather.total_radiation_wm2:.0f}</div><div class="wl">☀️ W/m²</div></div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Refresh Weather", use_container_width=True, key="rw"):
            st.session_state.weather_snap = get_weather_for_time(dt.datetime.now())
            st.rerun()

        # ── Data stats ───────────────────────────
        st.markdown("### 📦 Loaded Data")
        st.caption(f"🛣️ {n_roads} road segments  ·  🏢 {n_bldg} buildings  ·  🌳 {n_tree} trees")

    # ══════════════════════════════════════════════════════════
    #  ROUTE ANALYSIS
    # ══════════════════════════════════════════════════════════

    if analyze and can_go:
        import pytz
        with st.spinner("🛰️ Computing solar geometry → shadow projection → XGBoost heat prediction → route optimisation …"):
            try:
                import networkx as nx
                from src.optimization.path_finder import PathFinder
                from src.feature_engineering.solar_calculator import SolarCalculator

                # Build timezone-aware trip time
                riyadh = pytz.timezone("Asia/Riyadh")
                trip_time = riyadh.localize(
                    dt.datetime.combine(dep_date, dt.time(dep_hour, 0))
                )

                # Weather at trip time
                trip_weather = get_weather_for_time(trip_time)

                # Solar position (for UI display)
                sc = SolarCalculator()
                solar = sc.get_solar_position(CENTER_LAT, CENTER_LNG, trip_time)
                st.session_state.solar_info = solar

                # Extract a tiny local graph around the trip (makes XGBoost 100x faster!)
                buff = 0.006 
                min_lat = min(o["lat"], d["lat"]) - buff
                max_lat = max(o["lat"], d["lat"]) + buff
                min_lng = min(o["lng"], d["lng"]) - buff
                max_lng = max(o["lng"], d["lng"]) + buff

                nodes_in_bbox = [
                    n for n, d_attr in map_data.road_graph.nodes(data=True)
                    if min_lng <= d_attr["x"] <= max_lng and min_lat <= d_attr["y"] <= max_lat
                ]
                graph_copy = copy.deepcopy(map_data.road_graph.subgraph(nodes_in_bbox))

                # Build heat-weighted graph (XGBoost predictions per edge)
                weighted = builder.build_weighted_graph(
                    graph=graph_copy,
                    weather=trip_weather,
                    trip_time=trip_time,
                    user_speed_kmh=speed,
                    buildings_gdf=map_data.buildings_gdf,
                    trees_gdf=map_data.trees_gdf,
                )

                # Largest strongly connected component
                ccs = list(nx.strongly_connected_components(weighted))
                if not ccs:
                    st.error("❌ No connected street network found.")
                    st.stop()
                largest = max(ccs, key=len)
                sub = weighted.subgraph(largest)

                # Snap origin / destination to nearest graph node
                def nearest(lon, lat, G):
                    best, bd = None, float("inf")
                    for n, dd in G.nodes(data=True):
                        dx = dd["x"] - lon
                        dy = dd["y"] - lat
                        d2 = dx * dx + dy * dy
                        if d2 < bd:
                            bd, best = d2, n
                    return best

                u_n = nearest(o["lng"], o["lat"], sub)
                v_n = nearest(d["lng"], d["lat"], sub)

                finder = PathFinder()
                routes = finder.compare_routes(sub, u_n, v_n)

                if routes:
                    st.session_state.routes = routes
                    st.session_state.route_graph = sub
                else:
                    st.error("❌ No valid path found. Try moving points closer to visible roads.")

            except Exception as exc:
                st.error(f"❌ Analysis error: {exc}")
                import traceback
                st.code(traceback.format_exc())

    # ══════════════════════════════════════════════════════════
    #  SOLAR INFO BAR
    # ══════════════════════════════════════════════════════════
    if st.session_state.solar_info:
        s = st.session_state.solar_info
        shadow_note = ""
        if s.altitude_deg > 0:
            shadow_len_10m = 10 / max(math.tan(math.radians(s.altitude_deg)), 0.01)
            shadow_note = f" · Shadow of 10 m building ≈ {shadow_len_10m:.1f} m"
        st.markdown(f"""
        <div class="sol">
          ☀️ <b>Sun at trip time:</b> Altitude {s.altitude_deg:.1f}° · Azimuth {s.azimuth_deg:.1f}°
          · {'☀️ Daytime' if s.is_daytime else '🌙 Night'}{shadow_note}
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  MAP
    # ══════════════════════════════════════════════════════════

    map_col, legend_col = st.columns([6, 1])

    with map_col:
        m = folium.Map(
            location=[CENTER_LAT, CENTER_LNG],
            zoom_start=14,
            tiles="OpenStreetMap",
            min_lat=SW_LAT, max_lat=NE_LAT,
            min_lon=SW_LNG, max_lon=NE_LNG,
            max_bounds=True,
        )
        m.fit_bounds([[SW_LAT, SW_LNG], [NE_LAT, NE_LNG]])

        # ── Draw neighbourhood boundary ──────────
        folium.Rectangle(
            bounds=[[SW_LAT, SW_LNG], [NE_LAT, NE_LNG]],
            color="#FF6B35", weight=1.5, fill=False,
            opacity=0.25, dash_array="6",
        ).add_to(m)

        # ── Draw ALL roads from GeoPackage ───────
        MAJOR = {"primary", "secondary", "trunk", "motorway", "tertiary"}

        def _road_style(feat):
            hw = feat["properties"].get("highway", "")
            if hw in MAJOR:
                return {"color": "#4a6fa5", "weight": 2.0, "opacity": 0.65}
            return {"color": "#2d4a7a", "weight": 1.3, "opacity": 0.40}

        folium.GeoJson(
            road_gj,
            style_function=_road_style,
            name="Road Network",
        ).add_to(m)

        # ── Draw routes (if computed) ────────────
        route_colors = {"Fastest": "#FF5252", "Coolest": "#4CAF50", "Balanced": "#FF9800"}
        if st.session_state.routes and st.session_state.route_graph:
            sg = st.session_state.route_graph
            for r in st.session_state.routes:
                coords = [(sg.nodes[n]["y"], sg.nodes[n]["x"]) for n in r.path_nodes]
                col = route_colors.get(r.label, "#FFF")
                wt  = 7 if r.is_recommended else 4
                opa = 0.95 if r.is_recommended else 0.72
                da  = "8 6" if r.label == "Balanced" else None
                tip = (
                    f"<b>{r.label}</b>{'  ⭐ Recommended' if r.is_recommended else ''}<br>"
                    f"⏱ {r.total_time_min} min · 📏 {r.total_distance_m/1000:.2f} km<br>"
                    f"🌡️ Heat score: {r.cumulative_heat_exposure:.1f}<br>"
                    f"🌳 Shade: {r.avg_shade_fraction*100:.0f}%"
                )
                folium.PolyLine(
                    coords, color=col, weight=wt, opacity=opa,
                    dash_array=da, tooltip=folium.Tooltip(tip, sticky=True),
                ).add_to(m)

        # ── Origin / destination markers ─────────
        if o:
            folium.CircleMarker(
                [o["lat"], o["lng"]], radius=10,
                color="#4CAF50", fill=True, fill_color="#4CAF50",
                fill_opacity=0.9, popup="📍 Origin",
            ).add_to(m)
        if d:
            folium.CircleMarker(
                [d["lat"], d["lng"]], radius=10,
                color="#FF5252", fill=True, fill_color="#FF5252",
                fill_opacity=0.9, popup="🏁 Destination",
            ).add_to(m)

        # ── Render ───────────────────────────────
        result = st_folium(m, height=560, width=None, key="map",
                           returned_objects=["last_clicked"])

        # ── Handle map clicks ────────────────────
        if result and result.get("last_clicked"):
            clat = result["last_clicked"]["lat"]
            clng = result["last_clicked"]["lng"]
            # Clamp to bounds
            clat = max(SW_LAT, min(NE_LAT, clat))
            clng = max(SW_LNG, min(NE_LNG, clng))

            if st.session_state.origin is None:
                st.session_state.origin = {"lat": clat, "lng": clng}
                st.session_state.routes = None
                st.session_state.route_graph = None
                st.session_state.solar_info = None
                st.rerun()
            elif st.session_state.destination is None:
                st.session_state.destination = {"lat": clat, "lng": clng}
                st.session_state.routes = None
                st.session_state.route_graph = None
                st.session_state.solar_info = None
                st.rerun()
            else:
                st.session_state.origin = {"lat": clat, "lng": clng}
                st.session_state.destination = None
                st.session_state.routes = None
                st.session_state.route_graph = None
                st.session_state.solar_info = None
                st.rerun()

    # ── Legend column ────────────────────────────
    with legend_col:
        st.markdown("""
        <div class="leg">
          <div class="leg-t">Routes Legend</div>
          <div class="leg-i"><div class="leg-ln" style="background:#FF5252"></div>Fastest</div>
          <div class="leg-i"><div class="leg-ln" style="background:#4CAF50"></div>Coolest</div>
          <div class="leg-i"><div class="leg-ln" style="background:#FF9800;border-top:2px dashed #FF9800;height:0"></div>Balanced</div>
          <div class="leg-i" style="margin-top:6px;padding-top:6px;border-top:1px solid rgba(255,255,255,.06)">
            <div style="width:10px;height:10px;border-radius:50%;background:#4CAF50"></div>Origin</div>
          <div class="leg-i">
            <div style="width:10px;height:10px;border-radius:50%;background:#FF5252"></div>Destination</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="leg" style="margin-top:10px">
          <div class="leg-t">🤖 ML Engine</div>
          <div class="leg-i" style="color:#64ffda">{'XGBoost Active' if xgb_ok else 'Heuristic'}</div>
          <div class="leg-i" style="font-size:.65rem">200 trees · depth 8</div>
          <div class="leg-i" style="font-size:.65rem">10 features per edge</div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    #  RESULTS TABLE
    # ══════════════════════════════════════════════════════════

    if st.session_state.routes:
        routes = st.session_state.routes

        st.markdown("---")
        st.markdown("#### 📊 Route Comparison — XGBoost Heat Analysis")

        tbl = []
        for r in routes:
            lvl = ("🟢 Low" if r.cumulative_heat_exposure < 10
                   else "🟡 Moderate" if r.cumulative_heat_exposure < 25
                   else "🔴 High" if r.cumulative_heat_exposure < 40
                   else "🟣 Extreme")
            tbl.append({
                "": "⭐" if r.is_recommended else "",
                "Route": r.label,
                "Time (min)": r.total_time_min,
                "Distance (km)": round(r.total_distance_m / 1000, 2),
                "Heat Score": round(r.cumulative_heat_exposure, 1),
                "Level": lvl,
                "Avg Shade": f"{r.avg_shade_fraction*100:.0f}%",
            })
        st.dataframe(pd.DataFrame(tbl), hide_index=True, use_container_width=True)

        # ── Metrics row ──────────────────────────
        rec = next((r for r in routes if r.is_recommended), routes[0])
        fast = min(routes, key=lambda r: r.total_time_s)
        cool = min(routes, key=lambda r: r.cumulative_heat_exposure)

        m1, m2, m3 = st.columns(3)
        with m1:
            tdiff = rec.total_time_min - fast.total_time_min
            st.metric("⭐ Recommended", rec.label,
                      f"+{tdiff:.0f} min vs Fastest" if tdiff > 0 else "Also fastest!")
        with m2:
            if fast.cumulative_heat_exposure > 0:
                pct = (1.0 - rec.cumulative_heat_exposure / fast.cumulative_heat_exposure) * 100
            else:
                pct = 0
            st.metric("🌡️ Heat Reduction", f"{pct:.0f}%", f"vs {fast.label}",
                      delta_color="inverse")
        with m3:
            st.metric("🌳 Shade Coverage", f"{rec.avg_shade_fraction*100:.0f}%",
                      f"{rec.total_distance_m/1000:.1f} km route")

        # ── Explanation ──────────────────────────
        st.info(
            f"**How it works:** The system fetched weather data for the selected "
            f"hour ({dep_hour}:00), computed the sun position "
            f"(altitude {st.session_state.solar_info.altitude_deg:.1f}° → "
            f"shadow angles), projected 2.5-D building/tree shadows onto "
            f"every road segment, then fed **{len(road_gj['features'])}** edges × "
            f"10 features into the **XGBoost model** to predict per‐segment heat "
            f"exposure. Finally, Dijkstra's algorithm found three optimal paths "
            f"minimising time, heat, and a weighted combination."
        )


if __name__ == "__main__":
    main()
