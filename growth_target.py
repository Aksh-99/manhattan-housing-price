
import pandas as pd
 
import config
 
 
def build_neighborhood_year_table(df: pd.DataFrame) -> pd.DataFrame:
    neighborhood_year = df.groupby(["NEIGHBORHOOD", "SALE_YEAR"]).agg(
        MEDIAN_PRICE=("SALE_PRICE", "median"),
        SALES_COUNT=("SALE_PRICE", "size"),
        PCT_CONDO=("BUILDING_CLASS_CATEGORY", lambda x: x.str.contains("CONDO").mean()),
    ).reset_index()
 
    neighborhood_year = neighborhood_year[
        neighborhood_year["SALES_COUNT"] >= config.MIN_SALES_PER_NEIGHBORHOOD_YEAR
    ].copy()
 
    neighborhood_year["PRICE_SPIKE_YEAR"] = _flag_price_spikes(neighborhood_year)
    return neighborhood_year
 
 
def _flag_price_spikes(neighborhood_year: pd.DataFrame) -> pd.Series:
    pivot = neighborhood_year.pivot(index="NEIGHBORHOOD", columns="SALE_YEAR", values="MEDIAN_PRICE")
    spike_years = set()
 
    for neighborhood in pivot.index:
        row = pivot.loc[neighborhood]
        for year in row.index[:-1]:
            if year not in row.index or year + 1 not in row.index or (year - 1) not in row.index:
                continue
            prev_p, this_p, next_p = row.get(year - 1), row[year], row.get(year + 1)
            if pd.isna(prev_p) or pd.isna(this_p) or pd.isna(next_p):
                continue
            jump = (this_p - prev_p) / prev_p if prev_p else 0
            reversion = (next_p - this_p) / this_p if this_p else 0
            if jump > config.PRICE_SPIKE_JUMP_THRESHOLD and reversion < config.PRICE_SPIKE_REVERSION_THRESHOLD:
                spike_years.add((neighborhood, year))
 
    return neighborhood_year.apply(
        lambda r: (r["NEIGHBORHOOD"], r["SALE_YEAR"]) in spike_years, axis=1
    )
 
 
def build_growth_target(neighborhood_year: pd.DataFrame, forward_years: int = config.FORWARD_YEARS) -> pd.DataFrame:
    """For each neighborhood-year, compute the % price change `forward_years`
    ahead. Features (SALES_COUNT, PCT_CONDO, PRICE_SPIKE_YEAR) are carried
    from the BASE year, since those are what would be known at prediction time."""
    pivot = neighborhood_year.pivot(index="NEIGHBORHOOD", columns="SALE_YEAR", values="MEDIAN_PRICE")
    ny_indexed = neighborhood_year.set_index(["NEIGHBORHOOD", "SALE_YEAR"])
 
    records = []
    for neighborhood in pivot.index:
        row = pivot.loc[neighborhood]
        for year in row.index:
            if year + forward_years not in row.index:
                continue
            if (neighborhood, year) not in ny_indexed.index:
                continue
            base = ny_indexed.loc[(neighborhood, year)]
            p0, p1 = row[year], row[year + forward_years]
            if pd.isna(p0) or pd.isna(p1) or p0 == 0:
                continue
            records.append({
                "NEIGHBORHOOD": neighborhood,
                "BASE_YEAR": year,
                "BASE_PRICE": p0,
                "SALES_COUNT": base["SALES_COUNT"],
                "PCT_CONDO": base["PCT_CONDO"],
                "PRICE_SPIKE_YEAR": base["PRICE_SPIKE_YEAR"],
                f"GROWTH_RATE_{forward_years}YR": (p1 - p0) / p0,
            })
 
    return pd.DataFrame(records)
 
 
def add_price_momentum(growth_df: pd.DataFrame, neighborhood_year: pd.DataFrame,
                        window_years: int = config.MOMENTUM_WINDOW_YEARS) -> pd.DataFrame:
    """Add trailing price momentum (% change over the `window_years` before
    each BASE_YEAR) - a leading indicator of where a neighborhood was already
    heading before the prediction window starts."""
    pivot = neighborhood_year.pivot(index="NEIGHBORHOOD", columns="SALE_YEAR", values="MEDIAN_PRICE")
 
    def _momentum(row):
        n, year = row["NEIGHBORHOOD"], row["BASE_YEAR"]
        if n not in pivot.index or year - window_years not in pivot.columns or year not in pivot.columns:
            return None
        p_early, p_late = pivot.loc[n].get(year - window_years), pivot.loc[n].get(year)
        if pd.isna(p_early) or pd.isna(p_late) or p_early == 0:
            return None
        return (p_late - p_early) / p_early
 
    growth_df = growth_df.copy()
    growth_df[f"PRICE_MOMENTUM_{window_years}YR"] = growth_df.apply(_momentum, axis=1)
    return growth_df
 
 
def add_crisis_year_flag(growth_df: pd.DataFrame) -> pd.DataFrame:
    """Flags whether BASE_YEAR falls in a known market-wide crisis period
    (see config.CRISIS_YEARS). Gives the model at least two historical
    examples of "this is a crisis year" as a general, learnable pattern,
    rather than being blindsided by e.g. COVID as a complete unknown."""
    growth_df = growth_df.copy()
    growth_df["IS_CRISIS_YEAR"] = growth_df["BASE_YEAR"].isin(config.CRISIS_YEARS).astype(int)
    return growth_df
 
 
def fetch_mortgage_rate_by_year() -> pd.DataFrame:
    """Pull 30-year fixed mortgage rate history from FRED (free CSV endpoint,
    no API key) and average to one value per calendar year. Rate environment
    drives real-estate cycles market-wide - a feature the model otherwise has
    zero visibility into."""
    import requests
 
    resp = requests.get(config.FRED_MORTGAGE_RATE_URL, timeout=30)
    resp.raise_for_status()
    raw = pd.read_csv(pd.io.common.StringIO(resp.text))
    raw.columns = ["DATE", "MORTGAGE30US"]
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    raw["YEAR"] = raw["DATE"].dt.year
    raw["MORTGAGE30US"] = pd.to_numeric(raw["MORTGAGE30US"], errors="coerce")
 
    by_year = raw.groupby("YEAR")["MORTGAGE30US"].mean().reset_index()
    by_year.columns = ["BASE_YEAR", "AVG_MORTGAGE_RATE"]
    return by_year
 
 
def add_log_target(model_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add a log-transformed version of the growth-rate target. Growth rate
    is right-skewed (a handful of extreme spikes like a single new
    development's launch year can otherwise dominate model training) -
    log1p on the signed value stabilizes this while preserving direction."""
    import numpy as np
 
    model_df = model_df.copy()
    model_df[f"LOG_{target_col}"] = np.sign(model_df[target_col]) * np.log1p(np.abs(model_df[target_col]))
    return model_df
 
 
def aggregate_spatial_features(spatial_sales_path: str = config.SALES_WITH_SPATIAL_CSV) -> pd.DataFrame:
    """Average each sale-level spatial feature up to neighborhood-year level,
    so it can be joined onto the growth-rate target table."""
    spatial_df = pd.read_csv(spatial_sales_path, low_memory=False)
    spatial_df["NEIGHBORHOOD"] = spatial_df["NEIGHBORHOOD"].str.strip()
 
    dist_cols = [c for c in spatial_df.columns if c.startswith("DIST_NEAREST_")]
    count_cols = [c for c in spatial_df.columns if c.startswith("COUNT_") and c.endswith("_1KM")]
 
    agg_dict = {f"AVG_{c.replace('DIST_NEAREST_', 'DIST_').replace('_KM', '')}": (c, "mean") for c in dist_cols}
    agg_dict.update({f"AVG_{c}": (c, "mean") for c in count_cols})
 
    spatial_agg = spatial_df.groupby(["NEIGHBORHOOD", "SALE_YEAR"]).agg(**agg_dict).reset_index()
    spatial_agg = spatial_agg.rename(columns={"SALE_YEAR": "BASE_YEAR"})
    return spatial_agg
 
 
if __name__ == "__main__":
    from data_cleaning import load_and_clean
 
    df = load_and_clean()
    neighborhood_year = build_neighborhood_year_table(df)
    growth_df = build_growth_target(neighborhood_year)
    growth_df = add_price_momentum(growth_df, neighborhood_year)
    print("Growth-rate target rows:", growth_df.shape)
 
    spatial_agg = aggregate_spatial_features()
    spatial_agg.to_csv(config.SPATIAL_AGG_CSV, index=False)
 
    model_df = growth_df.merge(spatial_agg, on=["NEIGHBORHOOD", "BASE_YEAR"], how="left")
 
    mortgage_by_year = fetch_mortgage_rate_by_year()
    model_df = model_df.merge(mortgage_by_year, on="BASE_YEAR", how="left")
 
    target_col = f"GROWTH_RATE_{config.FORWARD_YEARS}YR"
    required_cols = [c for c in model_df.columns if c.startswith("AVG_")] + [
        f"PRICE_MOMENTUM_{config.MOMENTUM_WINDOW_YEARS}YR", "AVG_MORTGAGE_RATE"
    ]
    model_df = model_df.dropna(subset=required_cols)
    model_df = add_log_target(model_df, target_col)
 
    model_df.to_csv(config.GROWTH_MODEL_DATASET_CSV, index=False)
    print(f"Saved -> {config.GROWTH_MODEL_DATASET_CSV} ({model_df.shape})")
 