# AI Starter Project (Train + Inference API + Docker)

This starter provides:
- A reproducible training script (`train.py`)
- A FastAPI inference service (`app.py`)
- A Docker image for deployment (`Dockerfile`)

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Train and export model artifact

```bash
python train.py
```

This creates:
- `model/model.joblib`
- `model/model.metrics.json`

## 3) Run inference API locally

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[5.1,3.5,1.4,0.2]}'
```

## 4) Build and run with Docker

```bash
docker build -t ai-starter .
docker run --rm -p 8000:8000 ai-starter
```

If you trained the model outside Docker, mount it:

```bash
docker run --rm -p 8000:8000 -v "$(pwd)/model:/app/model" ai-starter
```
