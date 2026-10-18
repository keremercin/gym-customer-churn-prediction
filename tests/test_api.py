from fastapi.testclient import TestClient

from gym_churn.api.main import app
from gym_churn.train import train_and_eval


def test_health() -> None:
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["meta"]["app_version"] == "0.5.0"
    assert r.json()["meta"]["latency_ms"] is None
    assert "model_version" not in r.json()["meta"]


def test_version() -> None:
    c = TestClient(app)
    r = c.get("/version")
    assert r.status_code == 200
    assert r.json()["data"]["version"] == "0.5.0"


def test_predict(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    train_and_eval(data_path=str(repo_root / "data" / "gym_churn_us.csv"))
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
    assert "churn_class_prediction" in body
    assert "churn_score" in body
    assert "churn_probability" not in body
