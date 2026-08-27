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
        st.subheader("🗺️ Route Optimization")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📍 Origin**")
            orig_lat = st.number_input("Origin Lat", value=24.7136, format="%.4f", key="orig_lat")
            orig_lon = st.number_input("Origin Lon", value=46.6753, format="%.4f", key="orig_lon")

        with col2:
            st.markdown("**🏁 Destination**")
            dest_lat = st.number_input("Dest Lat", value=24.7000, format="%.4f", key="dest_lat")
            dest_lon = st.number_input("Dest Lon", value=46.6900, format="%.4f", key="dest_lon")

        if st.button("🔍 Find Routes", type="primary"):
            with st.spinner("Analyzing routes and heat conditions..."):
                st.info(
                    "🏗️ **Full route optimization requires OSMnx graph download.**\n\n"
                    "This is a demonstration of the system architecture. "
                    "In production, the system would:\n"
                    "1. Download the road network from OpenStreetMap\n"
                    "2. Fetch real-time weather data\n"
                    "3. Calculate solar position and shadows\n"
                    "4. Predict heat exposure for each road segment\n"
                    "5. Find optimal routes using weighted graph algorithms\n"
                    "6. Generate recommendations with explanations"
                )

                # Show example comparison
                st.markdown("---")
                st.subheader("📊 Example Route Comparison")

                example_data = pd.DataFrame({
                    "Route": ["🅰️ Fastest", "🅱️ Balanced ⭐", "🅲️ Coolest"],
                    "Time (min)": [15, 18, 22],
                    "Distance (km)": [3.2, 3.8, 4.1],
                    "Avg Heat Exposure": [42.5, 31.2, 25.8],
                    "Shade Coverage": ["12%", "45%", "68%"],
                    "Recommended": ["", "✓", ""],
                })
                st.dataframe(example_data, hide_index=True, use_container_width=True)

                # Heat exposure bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        name="Heat Exposure",
                        x=["Route A\n(Fastest)", "Route B\n(Balanced)", "Route C\n(Coolest)"],
                        y=[42.5, 31.2, 25.8],
                        marker_color=["#FF5252", "#FF9800", "#4CAF50"],
                    )
                ])
                fig.update_layout(
                    title="Heat Exposure Comparison",
                    yaxis_title="Heat Exposure Score",
                    template="plotly_dark",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

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
