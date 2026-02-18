from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from gym_churn.data import load_dataset, split_xy
from gym_churn.preprocess import make_preprocessor


def train_and_eval(
    data_path: str = "data/gym_churn_us.csv",
    model_out: str = "models/best_model.joblib",
    metrics_out: str = "reports/metrics.csv",
):
    df = load_dataset(data_path)
    x, y = split_xy(df)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=250, random_state=42),
    }

    rows = []
    best_name = None
    best_f1 = -1
    best_pipe = None

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", make_preprocessor()),
            ("model", model),
        ])
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_val)
        prob = pipe.predict_proba(x_val)[:, 1]

        acc = float(accuracy_score(y_val, pred))
        f1 = float(f1_score(y_val, pred))
        auc = float(roc_auc_score(y_val, prob))

        rows.append({"model": name, "accuracy": acc, "f1": f1, "roc_auc": auc})

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipe = pipe

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).sort_values("f1", ascending=False).to_csv(metrics_out, index=False)
    joblib.dump(best_pipe, model_out)

    return {"best_model": best_name, "best_f1": best_f1, "metrics_path": metrics_out, "model_path": model_out}


def predict(payload: dict, model_path: str = "models/best_model.joblib") -> dict:
    model = joblib.load(model_path)
    x = pd.DataFrame([payload])
    p = float(model.predict_proba(x)[0][1])
    return {"will_churn": bool(p >= 0.5), "churn_probability": round(p, 4)}
