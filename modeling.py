"""
Train and compare models for predicting neighborhood-level growth rate.

Compares Ridge, RandomForest, and XGBoost under two validation schemes:
  - Random split (the naive approach) - overstates real-world performance
    because it lets the model see data adjacent in time to the test rows.
  - Time-based split (train on early years, test on later years) - the
    validation scheme that reflects how the model would actually be used.

In testing, RandomForest was the most robust model on this dataset (~700
neighborhood-year rows): its R^2 held up reasonably under the time-based
split, while XGBoost consistently underperformed even after heavy
regularization (n_estimators, max_depth, reg_alpha/reg_lambda, subsample
tuning) - most likely because its extra flexibility has little to work
with at this sample size. RandomForest is used as the primary model
(see notebook for the full comparison and tuning attempts).
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import config

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


@dataclass
class SplitResult:
    model_name: str
    split_type: str
    mae: float
    r2: float


def get_model_configs() -> dict:
    configs = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42),
    }
    if HAS_XGB:
        configs["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
        )
    return configs


def evaluate_models(X_train, X_test, y_train, y_test, split_label: str) -> list[SplitResult]:
    results = []
    for name, model in get_model_configs().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f"  [{split_label}] {name}: MAE={mae:.4f}, R2={r2:.4f}")
        results.append(SplitResult(name, split_label, mae, r2))
    return results


def compare_random_vs_time_split(model_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    X = model_df.drop(columns=[target_col, "NEIGHBORHOOD"])
    y = model_df[target_col]

    print("=== RANDOM SPLIT (naive) ===")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
    random_results = evaluate_models(X_train_r, X_test_r, y_train_r, y_test_r, "random")

    print(f"\n=== TIME-BASED SPLIT (train BASE_YEAR < {config.TIME_SPLIT_YEAR}) ===")
    train_mask = model_df["BASE_YEAR"] < config.TIME_SPLIT_YEAR
    X_train_t, X_test_t = X[train_mask], X[~train_mask]
    y_train_t, y_test_t = y[train_mask], y[~train_mask]
    print(f"  Train rows: {len(X_train_t)}, Test rows: {len(X_test_t)}")
    time_results = evaluate_models(X_train_t, X_test_t, y_train_t, y_test_t, "time-based")

    all_results = random_results + time_results
    return pd.DataFrame([r.__dict__ for r in all_results])


def train_primary_model(model_df: pd.DataFrame, target_col: str) -> tuple:
    """Train the primary (RandomForest) model on the time-based train split.
    Returns (model, X_train, X_test, y_train, y_test) for downstream SHAP use."""
    X = model_df.drop(columns=[target_col, "NEIGHBORHOOD"])
    y = model_df[target_col]

    train_mask = model_df["BASE_YEAR"] < config.TIME_SPLIT_YEAR
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def explain_with_shap(model, X_test):
    """Returns a SHAP TreeExplainer + shap_values for the given model/test set."""
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values


if __name__ == "__main__":
    target_col = f"GROWTH_RATE_{config.FORWARD_YEARS}YR"
    model_df = pd.read_csv(config.GROWTH_MODEL_DATASET_CSV)

    comparison = compare_random_vs_time_split(model_df, target_col)
    print("\n", comparison)

    model, X_train, X_test, y_train, y_test = train_primary_model(model_df, target_col)
    print("\nPrimary model (RandomForest) trained on time-based split.")
