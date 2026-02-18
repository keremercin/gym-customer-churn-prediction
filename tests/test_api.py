from fastapi.testclient import TestClient

from gym_churn.api.main import app
from gym_churn.train import train_and_eval


def test_health() -> None:
    c = TestClient(app)
    assert c.get("/health").status_code == 200


def test_version() -> None:
    c = TestClient(app)
    r = c.get("/version")
    assert r.status_code == 200
    assert r.json()["data"]["version"] == "0.5.0"


def test_predict() -> None:
    train_and_eval()
    c = TestClient(app)
    payload = {
        "age": 31,
        "gender": "Male",
        "tenure_months": 3,
        "monthly_spend": 54.2,
        "visits_per_month": 3.0,
        "contract_type": "monthly",
        "support_tickets": 4,
    }
    r = c.post("/v1/predict", json=payload)
    assert r.status_code == 200
    body = r.json()["data"]
    assert "will_churn" in body
    assert "churn_probability" in body
