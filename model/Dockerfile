FROM python:3.10-slim
WORKDIR /app
COPY model.py model.pkl ./
RUN pip install flask numpy scikit-learn
CMD ["python", "model.py"]
