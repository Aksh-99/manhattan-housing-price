"""
Build the neighborhood-year forward growth-rate target variable.

Growth rate is computed at the NEIGHBORHOOD-YEAR level (not per-sale),
since the goal is predicting neighborhood-wide price trends, not individual
sale prices. Two data-quality issues are handled here, both found by
investigating outliers in the raw growth numbers rather than assuming a
threshold would catch them:

  - Sparse neighborhood-years (too few sales for a reliable median) are
    dropped via MIN_SALES_PER_NEIGHBORHOOD_YEAR.
  - "Price spike + reversion" years - a sharp one-year jump in median price
    followed by a sharp drop back down - are flagged via PRICE_SPIKE_YEAR.
    These are usually driven by a single new development's unit sales
    (e.g. one condo building's ~50 unit closings skewing a neighborhood's
    entire yearly median), not organic, sustained market movement. This
    was found to be a more robust detector than trying to flag it via
    address-level sales concentration, which struggled to distinguish a
    genuine new-building launch from neighborhoods that are structurally
    dominated by one large, steadily-turning-over co-op complex.
"""

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
    print("Growth-rate target rows:", growth_df.shape)

    spatial_agg = aggregate_spatial_features()
    spatial_agg.to_csv(config.SPATIAL_AGG_CSV, index=False)

    model_df = growth_df.merge(spatial_agg, on=["NEIGHBORHOOD", "BASE_YEAR"], how="left")
    model_df = model_df.dropna(subset=[c for c in model_df.columns if c.startswith("AVG_")])
    model_df.to_csv(config.GROWTH_MODEL_DATASET_CSV, index=False)
    print(f"Saved -> {config.GROWTH_MODEL_DATASET_CSV} ({model_df.shape})")
