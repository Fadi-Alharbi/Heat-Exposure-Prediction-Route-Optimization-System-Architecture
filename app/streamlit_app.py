"""
Streamlit Web Application
──────────────────────────
Interactive UI for the Heat Exposure Prediction & Route Optimization System.

Features:
  - Interactive map for origin/destination selection
  - Route comparison with heat exposure visualization
  - Departure time recommendations
  - SHAP feature importance display
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import datetime as dt
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config.settings import settings
from src.data_ingestion.weather_client import WeatherClient
from src.feature_engineering.heat_index_calculator import HeatIndexCalculator
from src.modeling.time_series_model import TimeSeriesModel
from src.explainability.route_comparator import RouteComparator

# ── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Heat Exposure Router | نظام التنبؤ الحراري",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
        direction: rtl;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #333;
    }
    .heat-low { color: #4CAF50; }
    .heat-moderate { color: #FF9800; }
    .heat-high { color: #F44336; }
    .heat-extreme { color: #9C27B0; }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""

    # ── Header ──────────────────────────────────────────────
    st.markdown('<div class="main-header">🌡️ Heat Exposure Router</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">نظام ذكي للتنبؤ بالتعرض الحراري وتحسين المسارات الخارجية</div>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        st.subheader("📍 Location")
        lat = st.number_input("Latitude", value=settings.DEFAULT_LATITUDE, format="%.4f")
        lon = st.number_input("Longitude", value=settings.DEFAULT_LONGITUDE, format="%.4f")

        st.subheader("🚲 Transport Mode")
        transport = st.selectbox(
            "Mode",
            ["🚲 Bike / Scooter", "🚶 Walking", "🚗 Driving"],
            index=0,
        )

        speed_map = {"🚲 Bike / Scooter": 15.0, "🚶 Walking": 5.0, "🚗 Driving": 40.0}
        speed = speed_map[transport]
        speed = st.slider("Speed (km/h)", 1.0, 60.0, speed, 0.5)

        st.subheader("⚖️ Optimization Preference")
        preference = st.select_slider(
            "Priority",
            options=["Fastest", "Slightly Fast", "Balanced", "Slightly Cool", "Coolest"],
            value="Balanced",
        )

        st.subheader("🕐 Departure Time")
        dep_date = st.date_input("Date", value=dt.date.today())
        dep_hour = st.slider("Hour", 0, 23, dt.datetime.now().hour)

    # ── Main Content ────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌤️ Current Conditions",
        "🗺️ Route Optimization",
        "⏰ Best Departure Time",
        "📊 About the System",
    ])

    # ── Tab 1: Current Weather & Heat ───────────────────────
    with tab1:
        st.subheader("Current Weather & Heat Conditions")

        if st.button("🔄 Fetch Weather Data", key="fetch_weather"):
            with st.spinner("Fetching weather data..."):
                try:
                    client = WeatherClient()
                    snapshot = client.get_current(lat, lon)

                    if snapshot:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("🌡️ Temperature", f"{snapshot.temperature_c}°C")
                        col2.metric("💧 Humidity", f"{snapshot.relative_humidity_pct}%")
                        col3.metric("💨 Wind", f"{snapshot.wind_speed_kmh} km/h")
                        col4.metric("☀️ Radiation", f"{snapshot.total_radiation_wm2:.0f} W/m²")

                        # Heat indices
                        calc = HeatIndexCalculator()
                        indices = calc.calculate_all(
                            snapshot.temperature_c,
                            snapshot.relative_humidity_pct,
                            snapshot.wind_speed_kmh,
                            snapshot.total_radiation_wm2,
                        )

                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Heat Index", f"{indices.heat_index_c:.1f}°C")
                        col2.metric("WBGT", f"{indices.wbgt_c:.1f}°C")
                        col3.metric("Apparent Temp", f"{indices.apparent_temperature_c:.1f}°C")

                        # Category badge
                        cat_colors = {
                            "comfortable": "🟢",
                            "caution": "🟡",
                            "danger": "🔴",
                            "extreme": "🟣",
                        }
                        emoji = cat_colors.get(indices.thermal_stress_category, "⚪")
                        st.info(f"{emoji} Thermal Stress Level: **{indices.thermal_stress_category.upper()}**")
                    else:
                        st.warning("Could not fetch weather data.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Tab 2: Route Optimization ───────────────────────────
    with tab2:
        st.subheader("🗺️ Interactive Route Optimizer")

        # Load Map Data via Cache
        @st.cache_resource
        def load_neighborhood_map_v2():
            from src.data_ingestion.osm_fetcher import OSMFetcher
            fetcher = OSMFetcher()
            gpkg = "planet_46.48,24.5401_46.5789,24.6004-geopackage/planet_46.48,24.5401_46.5789,24.6004.gpkg"
            return fetcher.fetch_from_geopackage(gpkg)

        map_data = load_neighborhood_map_v2()
        
        # Center of the neighborhood
        center_lat, center_lon = map_data.center if map_data.center != (0.0, 0.0) else (24.57, 46.53)

        if "origin" not in st.session_state:
            st.session_state.origin = None
        if "destination" not in st.session_state:
            st.session_state.destination = None

        st.markdown("**Step 1: Click on the map to set your ✨Origin📍. Step 2: Click to set your ✨Destination🏁**")

        import folium
        from streamlit_folium import st_folium

        m_interactive = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB dark_matter")
        
        if st.session_state.origin:
            folium.Marker(
                [st.session_state.origin["lat"], st.session_state.origin["lng"]], 
                popup="Origin 📍", icon=folium.Icon(color="green")
            ).add_to(m_interactive)
        
        if st.session_state.destination:
            folium.Marker(
                [st.session_state.destination["lat"], st.session_state.destination["lng"]], 
                popup="Destination 🏁", icon=folium.Icon(color="red")
            ).add_to(m_interactive)

        st_map = st_folium(m_interactive, height=400, width="100%", key="interactive_map")

        if st_map and st_map.get("last_clicked"):
            lat = st_map["last_clicked"]["lat"]
            lng = st_map["last_clicked"]["lng"]
            if st.session_state.origin is None:
                st.session_state.origin = {"lat": lat, "lng": lng}
                st.rerun()
            elif st.session_state.destination is None:
                st.session_state.destination = {"lat": lat, "lng": lng}
                st.rerun()
            else:
                st.session_state.origin = {"lat": lat, "lng": lng}
                st.session_state.destination = None
                st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.origin:
                st.success(f"📍 Origin: {st.session_state.origin['lat']:.4f}, {st.session_state.origin['lng']:.4f}")
            else:
                st.info("📍 Please click to select Origin")
                
        with col2:
            if st.session_state.destination:
                st.error(f"🏁 Destination: {st.session_state.destination['lat']:.4f}, {st.session_state.destination['lng']:.4f}")
            else:
                st.info("🏁 Please click to select Destination")

        with col3:
            if st.button("🔄 Reset Map Points"):
                st.session_state.origin = None
                st.session_state.destination = None
                st.rerun()

        if st.session_state.origin and st.session_state.destination:
            orig_lat, orig_lon = st.session_state.origin["lat"], st.session_state.origin["lng"]
            dest_lat, dest_lon = st.session_state.destination["lat"], st.session_state.destination["lng"]
            
            if st.button("🔍 Analyze & Find Best Routes", type="primary"):
                with st.spinner("Calculating sun position, building shadows, and route costs..."):
                    from src.optimization.graph_builder import GraphBuilder
                    from src.optimization.path_finder import PathFinder
                    import networkx as nx
                    from src.data_ingestion.weather_client import WeatherSnapshot

                    # Build Graph with Weather & Time
                    client = WeatherClient()
                    weather = client.get_current(center_lat, center_lon)
                    if weather is None:
                        weather = WeatherSnapshot(
                            timestamp=dt.datetime.now(), latitude=center_lat, longitude=center_lon,
                            temperature_c=40.0, relative_humidity_pct=20.0, wind_speed_kmh=10.0,
                            direct_radiation_wm2=700.0, diffuse_radiation_wm2=100.0, cloud_cover_pct=0.0
                        )
                    
                    trip_time = dt.datetime.combine(dep_date, dt.time(dep_hour, 0))
                    
                    builder = GraphBuilder()
                    graph = builder.build_weighted_graph(
                        graph=map_data.road_graph,
                        weather=weather,
                        trip_time=trip_time,
                        buildings_gdf=map_data.buildings_gdf,
                        trees_gdf=map_data.trees_gdf
                    )

                    # Extract largest strongly connected component to avoid "no path" errors
                    components = list(nx.strongly_connected_components(graph))
                    if not components:
                        st.error("The map does not contain connected streets.")
                        st.stop()
                    
                    largest_cc = max(components, key=len)
                    sub_graph = graph.subgraph(largest_cc)

                    def get_closest_node(lon, lat, G):
                        best_n = None
                        best_d = float('inf')
                        for n, data in G.nodes(data=True):
                            d = (data['x'] - lon)**2 + (data['y'] - lat)**2
                            if d < best_d:
                                best_d = d
                                best_n = n
                        return best_n

                    u_node = get_closest_node(orig_lon, orig_lat, sub_graph)
                    v_node = get_closest_node(dest_lon, dest_lat, sub_graph)

                    finder = PathFinder()
                    routes = finder.compare_routes(sub_graph, u_node, v_node)

                    if not routes:
                        st.error("Could not find a valid path. Try adjusting points closer to roads.")
                    else:
                        st.markdown("### 🗺️ Proposed Routes")
                        m_final = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="CartoDB dark_matter")
                        
                        colors = {"Fastest": "#FF5252", "Coolest": "#4CAF50", "Balanced": "#FF9800"}
                        
                        for r in routes:
                            coords = []
                            for node_id in r.path_nodes:
                                n_lat = sub_graph.nodes[node_id]['y']
                                n_lon = sub_graph.nodes[node_id]['x']
                                coords.append((n_lat, n_lon))
                                
                            color = colors.get(r.label, "#FFFFFF")
                            weight = 8 if r.is_recommended else 4
                            
                            folium.PolyLine(
                                coords, color=color, weight=weight, opacity=0.8,
                                tooltip=f"{r.label} | {r.total_time_min} mins | Est. Temperature: ~{weather.temperature_c + (r.cumulative_heat_exposure/10):.1f}°C"
                            ).add_to(m_final)

                        folium.Marker([orig_lat, orig_lon], popup="Origin 📍", icon=folium.Icon(color="green")).add_to(m_final)
                        folium.Marker([dest_lat, dest_lon], popup="Destination 🏁", icon=folium.Icon(color="red")).add_to(m_final)

                        from streamlit_folium import folium_static
                        folium_static(m_final, width=800, height=500)

                        st.markdown("### 📊 Detailed Route Comparison")
                        table_data = []
                        for r in routes:
                            table_data.append({
                                "Route 🛣️": f"{r.label} {'⭐ (Recommended)' if r.is_recommended else ''}",
                                "Time (min) ⏱️": r.total_time_min,
                                "Distance (km) 📍": round(r.total_distance_m / 1000, 2),
                                "Shade Coverage 🌳": f"{r.avg_shade_fraction*100:.1f}%",
                                "Heat Exposure Score 🌡️": r.cumulative_heat_exposure,
                            })
                        st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)

    # ── Tab 3: Best Departure Time ──────────────────────────
    with tab3:
        st.subheader("⏰ Best Departure Time Analysis")

        if st.button("📊 Analyze Departure Times"):
            with st.spinner("Computing hourly heat profiles..."):
                ts_model = TimeSeriesModel()
                profile = ts_model.forecast_daily_profile(base_temp_c=42.0)

                hours = [c.hour for c in profile]
                temps = [c.temperature_c for c in profile]
                exposures = [c.heat_exposure_estimate for c in profile]
                radiation = [c.solar_radiation_wm2 for c in profile]

                best_hour, best_exp = ts_model.find_best_departure_time(profile, 6, 20)

                # Recommendation
                st.success(
                    f"🕐 **Recommended Departure: {best_hour:02d}:00**\n\n"
                    f"Expected heat exposure: {best_exp:.1f} "
                    f"(lowest among {6}:00-20:00)"
                )

                # Charts
                col1, col2 = st.columns(2)

                with col1:
                    fig_temp = go.Figure()
                    fig_temp.add_trace(go.Scatter(
                        x=hours, y=temps,
                        mode='lines+markers',
                        name='Temperature',
                        line=dict(color='#FF6B35', width=2),
                    ))
                    fig_temp.update_layout(
                        title="Hourly Temperature Profile",
                        xaxis_title="Hour",
                        yaxis_title="Temperature (°C)",
                        template="plotly_dark",
                        height=350,
                    )
                    st.plotly_chart(fig_temp, use_container_width=True)

                with col2:
                    colors = ['#4CAF50' if e < 20 else '#FF9800' if e < 30 else '#FF5252'
                              for e in exposures]
                    fig_exp = go.Figure(data=[go.Bar(
                        x=hours, y=exposures,
                        marker_color=colors,
                    )])
                    fig_exp.update_layout(
                        title="Hourly Heat Exposure",
                        xaxis_title="Hour",
                        yaxis_title="Exposure Score",
                        template="plotly_dark",
                        height=350,
                    )
                    st.plotly_chart(fig_exp, use_container_width=True)

    # ── Tab 4: About ────────────────────────────────────────
    with tab4:
        st.subheader("📖 About the System")
        st.markdown("""
        ### نظام ذكي للتنبؤ بالتعرض الحراري وتحسين الرحلات الخارجية

        **Heat Exposure Prediction & Route Optimization System**

        ---

        #### 🎯 Goal
        Predict cumulative heat exposure during outdoor trips and optimize
        route + departure time to minimize thermal stress — especially for
        **delivery workers, cyclists, and scooter riders** in hot climates.

        #### 🏗️ Architecture Components

        | Layer | Components | Technologies |
        |-------|-----------|-------------|
        | **Data Ingestion** | Weather, Roads, Elevation, LST | Open-Meteo, OSMnx, MODIS |
        | **Feature Engineering** | Solar, Shadows, Surface, Heat Indices | pvlib, Shapely |
        | **ML Modeling** | Heat Exposure Prediction, Time Series | XGBoost, Prophet |
        | **Optimization** | Weighted Graph, Path Finding | NetworkX, Dijkstra/A* |
        | **Explainability** | SHAP, Route Comparison, Recommendations | SHAP, Custom |
        | **API** | REST Endpoints | FastAPI |
        | **UI** | Interactive Dashboard | Streamlit, Folium, Plotly |

        #### ⚠️ Disclaimer
        This system does **NOT** provide medical advice. It predicts heat
        exposure levels to help users make better-informed decisions about
        outdoor travel timing and routing.
        """)


if __name__ == "__main__":
    main()
