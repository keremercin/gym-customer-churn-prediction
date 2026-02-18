# gym-customer-churn-prediction

[![CI](https://github.com/keremercin/gym-customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/keremercin/gym-customer-churn-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![ML](https://img.shields.io/badge/ML-Churn%20Prediction-orange)

Production-style churn prediction pipeline with reproducible artifacts and API inference.

## Problem
Retention teams need calibrated churn probabilities to prioritize interventions.

## Architecture
- Dataset loading + feature split in `src/gym_churn/data.py`
- Preprocessing and model training in `src/gym_churn/train.py`
- FastAPI inference in `src/gym_churn/api/main.py`

See `docs/ARCHITECTURE.md`.

## Local Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python scripts/generate_sample_data.py
python scripts/train_model.py
python scripts/make_model_card.py
uvicorn gym_churn.api.main:app --reload --port 8700
```

## API Spec
- `GET /health`
- `GET /version`
- `POST /v1/predict`

Response envelope:
```json
{
  "status": "ok",
  "data": {},
  "meta": {"model_version": "0.5.0", "latency_ms": 0},
  "error": null
}
```

## Evaluation
```bash
python scripts/train_model.py
pytest
```

Outputs:
- `reports/metrics.csv`
- `reports/metrics.json`
- `reports/model_card.md`

## Results
Training pipeline persists best-performing model and publishes reproducible metrics artifacts.

## Limitations
- Single dataset baseline; no external validation split yet.
- No drift monitoring in current pipeline.
- Explainability output is model-card level only.

## Roadmap
- Add drift checks and threshold alerts.
- Add feature attribution summary.
- Add scheduled retraining workflow.

## Docs
- `docs/CASE_STUDY.md`
- `docs/ARCHITECTURE.md`
- `docs/DEMO_SCRIPT_90S.md`
