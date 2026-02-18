from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["age", "tenure_months", "monthly_spend", "visits_per_month", "support_tickets"]
CAT_COLS = ["gender", "contract_type"]


def make_preprocessor():
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num, NUM_COLS),
        ("cat", cat, CAT_COLS),
    ])
