from fastapi import FastAPI
from pydantic import BaseModel

from gym_churn.train import predict

app = FastAPI(title="Gym Churn Prediction API", version="0.2.0")


class PredictRequest(BaseModel):
    age: float
    gender: str
    tenure_months: float
    monthly_spend: float
    visits_per_month: float
    contract_type: str
    support_tickets: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "gym-churn-api"}


@app.post("/v1/predict")
def predict_churn(req: PredictRequest) -> dict:
    return predict(req.model_dump())
