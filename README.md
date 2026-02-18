# 🏋️ gym-customer-churn-prediction

[![CI](https://github.com/keremercin/gym-customer-churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/keremercin/gym-customer-churn-prediction/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![ML](https://img.shields.io/badge/ML-Churn%20Prediction-orange)

Production-style churn prediction project for gym subscriptions.

## Problem
Retention teams need probability-based churn risk estimates to trigger proactive interventions.

## Solution
- structured data preprocessing pipeline
- model comparison (`logistic_regression`, `random_forest`)
- best-model artifact saving
- FastAPI inference endpoint (`/v1/predict`)

---

## API
- `GET /health`
- `POST /v1/predict`

Swagger: `http://localhost:8700/docs`

Example request:
```json
{
  "age": 31,
  "gender": "Male",
  "tenure_months": 3,
  "monthly_spend": 54.2,
  "visits_per_month": 3.0,
  "contract_type": "monthly",
  "support_tickets": 4
}
```

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

python scripts/generate_sample_data.py
python scripts/train_model.py
python scripts/make_model_card.py

uvicorn gym_churn.api.main:app --reload --port 8700
```

---

## Structure

```text
src/gym_churn/
├─ api/main.py
├─ data.py
├─ preprocess.py
└─ train.py
```

---

## Quality
- tests + CI
- lint checks
- model card output in `reports/model_card.md`

---

## Docs
- `docs/CASE_STUDY.md`
