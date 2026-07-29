import numpy as np
import pandas as pd

import config


def normalize_whitespace(series: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace and collapse internal double-spaces."""
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def load_and_clean(path: str = config.RAW_SALES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    df["NEIGHBORHOOD"] = normalize_whitespace(df["NEIGHBORHOOD"])
    df["BUILDING_CLASS_CATEGORY"] = normalize_whitespace(df["BUILDING_CLASS_CATEGORY"])
    df["ADDRESS"] = df["ADDRESS"].astype(str).str.strip()
    # Base street address (drops unit/apartment suffix after the first comma) -
    # needed later for building-level grouping (e.g. new-development detection).
    df["BASE_ADDRESS"] = df["ADDRESS"].str.split(",").str[0].str.strip()

    df = df[df["SALE_PRICE"] >= config.MIN_SALE_PRICE]

    for neighborhood in config.EXCLUDED_NEIGHBORHOODS:
        df = df[df["NEIGHBORHOOD"] != neighborhood]

    df = df[df["BUILDING_CLASS_CATEGORY"].isin(config.RESIDENTIAL_BUILDING_CLASSES)]

    upper_cap = df["SALE_PRICE"].quantile(config.UPPER_PRICE_PERCENTILE)
    df = df[df["SALE_PRICE"] <= upper_cap]

    df = _handle_square_footage(df)
    df = _handle_year_built(df)

    return df


def _handle_square_footage(df: pd.DataFrame) -> pd.DataFrame:
    """Flag whether square footage is real vs. structurally missing, rather
    than imputing (imputation would fabricate values for the ~90%+ of rows -
    mostly condos/co-ops - where DOF's sales file doesn't carry unit-level
    square footage)."""
    df = df.copy()
    df["SQFT_KNOWN"] = (df["GROSS_SQUARE_FEET"] > 0).astype(int)
    df["GROSS_SQUARE_FEET"] = df["GROSS_SQUARE_FEET"].replace(0, np.nan)
    df["LAND_SQUARE_FEET"] = df["LAND_SQUARE_FEET"].replace(0, np.nan)
    return df


def _handle_year_built(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["YEAR_BUILT"] = df["YEAR_BUILT"].replace(0, np.nan)
    df.loc[df["YEAR_BUILT"] < 1800, "YEAR_BUILT"] = np.nan  # catches data-entry typos, e.g. "190"
    df["YEAR_BUILT"] = df.groupby("BUILDING_CLASS_CATEGORY")["YEAR_BUILT"].transform(
        lambda x: x.fillna(x.median())
    )
    return df


if __name__ == "__main__":
    df = load_and_clean()
    print("Cleaned dataset:", df.shape)
    df.to_csv(config.MODELING_READY_CSV, index=False)
    print(f"Saved -> {config.MODELING_READY_CSV}")
