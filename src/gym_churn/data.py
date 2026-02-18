from pathlib import Path

import pandas as pd

TARGET = "churn"


def load_dataset(path: str | Path = "data/gym_churn_us.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame):
    y = df[TARGET].astype(int)
    x = df.drop(columns=[TARGET])
    return x, y
