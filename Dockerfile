FROM python:3.12-slim
LABEL authors="wojtek"

WORKDIR /app
COPY app/__init__.py .
COPY app/app.py .
COPY app/predict.py .
COPY app/assets ./assets
COPY app/requirements.txt .
COPY app/model.joblib .

RUN pip3 install -r requirements.txt
EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "./app.py", "--server.address=0.0.0.0", "--server.port=8080"]