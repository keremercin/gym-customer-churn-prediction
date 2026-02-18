from gym_churn.train import train_and_eval


def test_training(tmp_path) -> None:
    model = tmp_path / "m.joblib"
    metrics = tmp_path / "m.csv"
    out = train_and_eval(model_out=str(model), metrics_out=str(metrics))
    assert out["best_model"] in {"logistic_regression", "random_forest"}
    assert model.exists()
    assert metrics.exists()
