FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY env/ ./env/
COPY grader/ ./grader/
COPY agent/ ./agent/
COPY data/raw/ ./data/raw/

RUN mkdir -p data/processed

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
