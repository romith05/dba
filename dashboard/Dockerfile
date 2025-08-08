# Use official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy Streamlit app
COPY streamlit.py ./

# Install system dependencies (for rasterio and pyproj)
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    libgdal-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL (required for rasterio and pyproj)
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Python dependencies
RUN pip install --no-cache-dir \
    streamlit \
    requests \
    folium \
    streamlit_folium \
    rasterio \
    pyproj \
    plotly \
    pandas \
    numpy

# Run Streamlit app
CMD ["streamlit", "run", "streamlit.py", "--server.port=8000", "--server.address=0.0.0.0"]
