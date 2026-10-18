import json
from pathlib import Path

from gym_churn.train import predict, train_and_eval


def test_training(tmp_path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "m.joblib"
    metrics = tmp_path / "m.csv"
    out = train_and_eval(
        data_path=str(repo_root / "data" / "gym_churn_us.csv"),
        model_out=str(model),
        metrics_out=str(metrics),
    )
    assert out["best_model"] in {"logistic_regression", "random_forest"}
    assert model.exists()
    assert metrics.exists()
    metadata = json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert metadata["app_version"] == "0.5.0"
    assert metadata["dataset_kind"] == "synthetic_generated"
    assert metadata["label_rule"] == "rule in scripts/generate_sample_data.py"
    assert metadata["evaluation_method"] == (
        "single stratified validation split; model selection and reporting use the same split"
    )
    assert "model_version" not in metadata


def test_predict_exposes_an_uncalibrated_churn_score(monkeypatch):
    class Model:
        def predict_proba(self, _frame):
            return [[0.2, 0.8]]

    monkeypatch.setattr("gym_churn.train.joblib.load", lambda _path: Model())
    result = predict({"visits_per_month": 2})

    assert result == {"churn_class_prediction": True, "churn_score": 0.8}


def test_tracked_report_identifies_metadata_only_snapshot():
    repo_root = Path(__file__).resolve().parents[1]
    metadata = json.loads((repo_root / "reports" / "metrics.json").read_text(encoding="utf-8"))
    benchmark = (repo_root / "reports" / "benchmark.md").read_text(encoding="utf-8")
    model_card = (repo_root / "reports" / "model_card.md").read_text(encoding="utf-8")

    assert metadata["artifact_status"] == "metadata_only"
    assert "metrics" not in metadata
    assert "best_model" not in metadata
    assert "No training metrics are included in this tracked snapshot" in benchmark
    assert "metadata-only" in model_card
