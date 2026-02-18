from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    p = Path("reports/metrics.csv")
    if not p.exists():
        raise SystemExit("Run training first")

    df = pd.read_csv(p)
    lines = [
        "# Model Card — Gym Churn Prediction",
        "",
        "| Model | Accuracy | F1 | ROC-AUC |",
        "|---|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |")

    Path("reports/model_card.md").write_text("\n".join(lines), encoding="utf-8")
    print("written reports/model_card.md")
