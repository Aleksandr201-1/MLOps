FROM python:3.8-slim

WORKDIR /app

RUN apt-get update
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5000

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "s3://mlflow-artifacts/", \
     "--host", "0.0.0.0"]
