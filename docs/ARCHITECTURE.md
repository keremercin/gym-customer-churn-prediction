# Architecture

## Components
- Training and model selection: `src/gym_churn/train.py`
- Preprocessing pipeline: `src/gym_churn/preprocess.py`
- Data utilities: `src/gym_churn/data.py`
- FastAPI inference: `src/gym_churn/api/main.py`

## Flow
1. Load and split churn dataset.
2. Train candidate models.
3. Persist best artifact and metrics outputs.
4. Serve predictions with standardized response envelope.
