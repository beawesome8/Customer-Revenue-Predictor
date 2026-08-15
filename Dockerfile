FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY model_artifact.joblib model_manifest.json ./

EXPOSE 8080
CMD ["uvicorn", "src.serve.main:app", "--host", "0.0.0.0", "--port", "8080"]
