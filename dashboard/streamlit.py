import streamlit as st
import folium
from streamlit_folium import st_folium
import random
import requests
import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from folium.plugins import HeatMap
import plotly.express as px
import pickle as pkl    
import pandas as pd
import numpy as np
from io import StringIO
def is_water_body(lat, lon, tiff_path):
    with rasterio.open(tiff_path) as src:
        # Set up coordinate transformer if needed
        if src.crs.to_string() != "EPSG:4326":
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            lon, lat = transformer.transform(lon, lat)
        
        try:
            row, col = src.index(lon, lat)
            value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
            return value == 1
        except Exception as e:
            print("Error:", e)
            return False

# Layout setup
st.set_page_config(layout="wide")
st.title("Wildfire Risk Prediction Dashboard")
st.markdown("---")

map_mode = st.radio("Select Map View:", ["Click Prediction Map", "Historical Heatmap"], horizontal=True)

if map_mode == "Click Prediction Map":
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.subheader("🗺️ Click to Predict Risk")

        m = folium.Map(location=[54.0, -115.0], zoom_start=6, min_zoom=6, max_bounds=True)

        # folium.Marker(
        #     location=[53.5, -113.5],
        #     tooltip="Click on the map",
        #     icon=folium.Icon(color="blue")
        # ).add_to(m)

        map_data = st_folium(m, width=1000, height=700)

    with right_col:
        st.subheader("📈 Prediction Result")

        if map_data and map_data.get("last_clicked"):
            lat = map_data["last_clicked"]["lat"]
            lon = map_data["last_clicked"]["lng"]

            st.markdown(f"**Selected Location:** `{lat:.4f}, {lon:.4f}`")

            if is_water_body(lat, lon,"dsw-2023-mask.tif"):
                percentage = 0.0
                is_water = True
            else:
                percentage = round(random.uniform(0, 1) * 100, 1)
                is_water = False

            st.markdown(
                f"""
                <div style="font-family: 'Georgia', serif; 
                            font-size: 48px; 
                            font-weight: bold; 
                            color: {"#000000" if is_water else "#000000"}; 
                            text-align: center;
                            margin-top: 30px;">
                    {percentage:.1f}% Risk
                </div>
                """,
                unsafe_allow_html=True
            )

            if is_water:
                st.markdown(
                    f"""
                    <div style="text-align: center; color: gray; font-style: italic;">
                        A water body
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("Click on the map to get wildfire risk prediction.")

# elif map_mode == "Historical Heatmap":
#     st.subheader("🔥 Wildfire Live Probability Heatmap")
#     df=pd.read_csv("grid_points_with_prob.csv")
#     df['vegetation_type'] = np.random.choice(['1', '11', '10'], size=len(df))
#     print(df.head())
#     m = folium.Map(location=[54.0, -114.0], zoom_start=5)
#     heat_data = df[['lat', 'lon','prob']].values.tolist()
#     HeatMap(heat_data, radius=15, blur=10, max_zoom=8).add_to(m)
#     st_folium(m, width=700, height=500)
# # Placeholder for the heatmap
#     st.markdown(
#         """        <div style="text-align: center; color: gray; font-style: italic;">
#             This is a live heatmap showing wildfire risk probabilities based on the latest model predictions.
#         </div>
#         """,
#         unsafe_allow_html=True
#     )       

# Bottom placeholder section
base_url = "http://duckdb:9999"
def build_query(viz_type, year=None, unit=None):
    if viz_type == "heatmap":
        return """
            SELECT latitude, longitude
            FROM my_table
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
    elif viz_type == "bar":
        return """
            SELECT fire_year, COUNT(*) as count
            FROM my_table
            WHERE fire_year IS NOT NULL
            GROUP BY fire_year
            ORDER BY fire_year
        """
    elif viz_type == "hist":
        return """
            SELECT size
            FROM my_table
            WHERE size IS NOT NULL
        """
    elif viz_type == "pie":
        return """
            SELECT general_cause_desc, COUNT(*) as count
            FROM my_table
            WHERE general_cause_desc IS NOT NULL
            GROUP BY general_cause_desc
            ORDER BY count DESC
        """
    else:
        return "SELECT 1"

# Layout
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# === 1. HEATMAP ===
with col1:
    st.subheader("🔥 Wildfire Heatmap (Density)")
    query = build_query("heatmap")
    response = requests.post(base_url, data=query, timeout=10)
    df_heat = pd.read_json(StringIO(response.text), lines=True)

    map_center = [54.0, -115.0]
    m = folium.Map(location=map_center, zoom_start=4, min_zoom=6, max_bounds=True)
    heat_data = df_heat[['latitude', 'longitude']].dropna().values.tolist()
    HeatMap(heat_data, radius=10).add_to(m)
    st_folium(m, width=700, height=700)

# === 2. BAR CHART ===
with col2:
    st.subheader("📊 Yearly Wildfire Counts")
    query = build_query("bar")
    response = requests.post(base_url, data=query, timeout=10)
    df_yearly = pd.read_json(StringIO(response.text), lines=True)
    st.bar_chart(df_yearly.set_index("fire_year"))

# === 3. HISTOGRAM ===
with col3:
    st.subheader("📏 Distribution of Fire Sizes")
    query = build_query("hist")
    response = requests.post(base_url, data=query, timeout=10)
    df_size = pd.read_json(StringIO(response.text), lines=True)
    st.plotly_chart(px.histogram(df_size, x="size", nbins=50, title="Histogram of Fire Sizes"))

# === 4. PIE CHART ===
with col4:
    st.subheader("🧯 Fire Causes")
    query = build_query("pie")
    response = requests.post(base_url, data=query, timeout=10)
    df_cause = pd.read_json(StringIO(response.text), lines=True)

    if not df_cause.empty:
        st.plotly_chart(px.pie(df_cause, names="general_cause_desc", values="count", title="Fire Causes"))
    else:
        st.info("No cause data found.")
