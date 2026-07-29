
from dataclasses import dataclass
 
import numpy as np
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
 
 
def _inverse_log_transform(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.expm1(np.abs(values))
 
 
def ensemble_predict(models: dict, X) -> np.ndarray:
    preds = np.column_stack([m.predict(X) for m in models.values()])
    return preds.mean(axis=1)
 
 
def three_way_split(model_df: pd.DataFrame, val_year: int, test_year: int):
    train_mask = model_df["BASE_YEAR"] < val_year
    val_mask = (model_df["BASE_YEAR"] >= val_year) & (model_df["BASE_YEAR"] < test_year)
    test_mask = model_df["BASE_YEAR"] >= test_year
    return model_df[train_mask], model_df[val_mask], model_df[test_mask]
 
 
def evaluate_ensemble(model_df: pd.DataFrame, target_col: str, val_year: int, test_year: int,
                        exclude_cols: list[str] | None = None, is_log_target: bool = False,
                        group_col: str = "NEIGHBORHOOD"):
    drop_cols = [target_col, group_col] + (exclude_cols or [])
    feature_cols = [c for c in model_df.columns if c not in drop_cols]
 
    train_df, val_df, test_df = three_way_split(model_df, val_year, test_year)
    print(f"Train: {len(train_df)} rows | Validation: {len(val_df)} rows | Test: {len(test_df)} rows")
 
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
 
    models = get_model_configs()
    for m in models.values():
        m.fit(X_train, y_train)
 
    for label, X_eval, y_eval in [("validation", X_val, y_val), ("held-out test", X_test, y_test)]:
        print(f"\n--- {label} set ---")
        for name, m in models.items():
            preds = m.predict(X_eval)
            r2 = r2_score(y_eval, preds)
            print(f"  {name}: R2={r2:.4f}")
 
        ensemble_preds = ensemble_predict(models, X_eval)
        r2_ens = r2_score(y_eval, ensemble_preds)
        print(f"  Ensemble (avg of all 3): R2={r2_ens:.4f}")
 
 
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
 
 
def evaluate_models(X_train, X_test, y_train, y_test, split_label: str,
                     is_log_target: bool = False) -> list[SplitResult]:
    results = []
    for name, model in get_model_configs().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)  # on whatever scale the model was trained on
 
        if is_log_target:
            mae = mean_absolute_error(_inverse_log_transform(y_test.values),
                                       _inverse_log_transform(preds))
        else:
            mae = mean_absolute_error(y_test, preds)
 
        print(f"  [{split_label}] {name}: MAE={mae:.4f}, R2={r2:.4f}")
        results.append(SplitResult(name, split_label, mae, r2))
    return results
 
 
def compare_random_vs_time_split(model_df: pd.DataFrame, target_col: str,
                                   is_log_target: bool = False,
                                   exclude_cols: list[str] | None = None) -> pd.DataFrame:
    drop_cols = [target_col, "NEIGHBORHOOD"] + (exclude_cols or [])
    X = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])
    y = model_df[target_col]
 
    print("=== RANDOM SPLIT (naive) ===")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
    random_results = evaluate_models(X_train_r, X_test_r, y_train_r, y_test_r, "random", is_log_target)
 
    print(f"\n=== TIME-BASED SPLIT (train BASE_YEAR < {config.TIME_SPLIT_YEAR}) ===")
    train_mask = model_df["BASE_YEAR"] < config.TIME_SPLIT_YEAR
    X_train_t, X_test_t = X[train_mask], X[~train_mask]
    y_train_t, y_test_t = y[train_mask], y[~train_mask]
    print(f"  Train rows: {len(X_train_t)}, Test rows: {len(X_test_t)}")
    time_results = evaluate_models(X_train_t, X_test_t, y_train_t, y_test_t, "time-based", is_log_target)
 
    all_results = random_results + time_results
    return pd.DataFrame([r.__dict__ for r in all_results])
 
 
def rolling_window_evaluate(model_df: pd.DataFrame, target_col: str, min_train_years: int = 8,
                              exclude_cols: list[str] | None = None, is_log_target: bool = False,
                              group_col: str = "NEIGHBORHOOD") -> pd.DataFrame:
    drop_cols = [target_col, group_col] + (exclude_cols or [])
    feature_cols = [c for c in model_df.columns if c not in drop_cols]
 
    years = sorted(model_df["BASE_YEAR"].unique())
    start_idx = min_train_years  # first year with enough prior years to train on
    results = []
 
    for test_year in years[start_idx:]:
        train_df = model_df[model_df["BASE_YEAR"] < test_year]
        test_df = model_df[model_df["BASE_YEAR"] == test_year]
 
        if len(test_df) == 0 or len(train_df) == 0:
            continue
 
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]
 
        model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
 
        if is_log_target:
            mae = mean_absolute_error(_inverse_log_transform(y_test.values), _inverse_log_transform(preds))
        else:
            mae = mean_absolute_error(y_test, preds)
 
        is_crisis = bool(test_df["IS_CRISIS_YEAR"].iloc[0]) if "IS_CRISIS_YEAR" in test_df.columns else None
        results.append({
            "test_year": test_year, "train_rows": len(train_df), "test_rows": len(test_df),
            "mae": mae, "r2": r2, "is_crisis_year": is_crisis,
        })
        print(f"  Test year {test_year} (train on {len(train_df)} rows, "
              f"test on {len(test_df)}): MAE={mae:.4f}, R2={r2:.4f}"
              + (f"  [CRISIS YEAR]" if is_crisis else ""))
 
    results_df = pd.DataFrame(results)
    print(f"\nMean R2 across all years: {results_df['r2'].mean():.4f}")
    if "is_crisis_year" in results_df.columns and results_df["is_crisis_year"].notna().any():
        print(f"Mean R2, crisis years only: {results_df[results_df['is_crisis_year']==True]['r2'].mean():.4f}")
        print(f"Mean R2, non-crisis years: {results_df[results_df['is_crisis_year']==False]['r2'].mean():.4f}")
    return results_df
 
 
def train_primary_model(model_df: pd.DataFrame, target_col: str,
                          exclude_cols: list[str] | None = None) -> tuple:
    drop_cols = [target_col, "NEIGHBORHOOD"] + (exclude_cols or [])
    X = model_df.drop(columns=[c for c in drop_cols if c in model_df.columns])
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
