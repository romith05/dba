FROM python:3.10-slim
WORKDIR /app
COPY model.py model.pkl ./
RUN pip install --no-cache-dir \
    flask \
    numpy \
    pandas \
    scikit-learn==1.3.2 \
    joblib \
    requests
CMD ["python", "model.py"]
