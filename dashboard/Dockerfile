FROM python:3.10
WORKDIR /app
COPY streamlit.py ./
RUN pip install streamlit requests folium streamlit_folium 
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
