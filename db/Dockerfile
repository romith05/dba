FROM python:3.10

#  Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

#  working directory
WORKDIR /app

#  DuckDB via pip
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN pip install --no-cache-dir duckdb

#  Parquet file
COPY ./wildfire_data.parquet ./wildfire_data.parquet

#  DuckDB server script
COPY ./dbserve.py ./dbserve.py

# Run 
CMD ["python3", "dbserve.py"]
