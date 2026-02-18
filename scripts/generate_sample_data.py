import numpy as np
import pandas as pd


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1200

    age = rng.integers(18, 66, size=n)
    gender = rng.choice(["Male", "Female"], size=n)
    tenure = rng.integers(1, 49, size=n)
    monthly_spend = rng.normal(75, 20, size=n).clip(20, 200)
    visits = rng.normal(8, 3, size=n).clip(0, 25)
    contract = rng.choice(["monthly", "quarterly", "annual"], size=n, p=[0.55, 0.30, 0.15])
    tickets = rng.poisson(1.4, size=n)

    risk = (
        0.35 * (contract == "monthly").astype(float)
        + 0.25 * (visits < 5).astype(float)
        + 0.20 * (tickets >= 3).astype(float)
        + 0.20 * (tenure < 6).astype(float)
    )
    churn = (risk + rng.normal(0, 0.12, size=n) > 0.45).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "tenure_months": tenure,
            "monthly_spend": monthly_spend.round(2),
            "visits_per_month": visits.round(2),
            "contract_type": contract,
            "support_tickets": tickets,
            "churn": churn,
        }
    )
    df.to_csv("data/gym_churn_us.csv", index=False)
    print("generated data/gym_churn_us.csv", df.shape)
