"""
STEP 2 of 2: Pull Manhattan residential building permits from NYC DOB
(free Socrata API), assign each permit to the nearest known neighborhood
using the same nearest-centroid approach as spatial_features.py, and
aggregate into a neighborhood-year permit-count feature - a genuine
leading indicator, since new construction/major-renovation activity
happening NOW plausibly precedes the price growth it's meant to predict,
unlike static amenities (a subway stop doesn't move) or price momentum
(a lagging, not leading, signal).

Only NB (New Building) and A1 (major Alteration) job types are counted -
see config.SIGNIFICANT_JOB_TYPES. A2/A3 cover minor work (a new
bathroom, a boiler swap) that isn't a meaningful gentrification signal
and would mostly add noise.

Schema confirmed via explore_permit_schema.py before writing this -
key fields: borough, issuance_date, gis_latitude, gis_longitude,
job_type, residential, zip_code.
"""

import time

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree

import config


def fetch_manhattan_permits(force_refresh: bool = False) -> pd.DataFrame:
    """Pull all Manhattan residential NB/A1 permits, paginated via
    $limit/$offset (Socrata caps results per request, same pattern as the
    Census batch geocoder)."""
    import os

    if not force_refresh and os.path.exists(config.DOB_PERMITS_CACHE):
        cached = pd.read_csv(config.DOB_PERMITS_CACHE, low_memory=False)
        print(f"Loaded cached permits: {len(cached)} rows")
        return cached

    job_type_filter = " OR ".join(f"job_type='{jt}'" for jt in config.SIGNIFICANT_JOB_TYPES)
    where_clause = f"borough='MANHATTAN' AND residential='YES' AND ({job_type_filter})"

    all_rows = []
    offset = 0
    while True:
        params = {
            "$where": where_clause,
            "$limit": config.DOB_PERMIT_CHUNK_SIZE,
            "$offset": offset,
            "$select": "issuance_date,gis_latitude,gis_longitude,job_type,zip_code",
        }
        resp = requests.get(config.DOB_PERMITS_URL, params=params, timeout=60)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        all_rows.extend(batch)
        print(f"  fetched {len(all_rows)} permits so far...")
        offset += config.DOB_PERMIT_CHUNK_SIZE
        if len(batch) < config.DOB_PERMIT_CHUNK_SIZE:
            break
        time.sleep(0.5)

    permits = pd.DataFrame(all_rows)
    permits.to_csv(config.DOB_PERMITS_CACHE, index=False)
    print(f"Saved {len(permits)} permits -> {config.DOB_PERMITS_CACHE}")
    return permits


def assign_permits_to_neighborhoods(permits: pd.DataFrame, neighborhood_centroids: pd.DataFrame) -> pd.DataFrame:
    """Assign each permit to its nearest neighborhood centroid (same
    nearest-neighbor approach as spatial_features.py's amenity matching).
    `neighborhood_centroids` must have columns: NEIGHBORHOOD, LAT, LON."""
    permits = permits.copy()
    permits["gis_latitude"] = pd.to_numeric(permits["gis_latitude"], errors="coerce")
    permits["gis_longitude"] = pd.to_numeric(permits["gis_longitude"], errors="coerce")
    permits = permits.dropna(subset=["gis_latitude", "gis_longitude"])

    centroid_coords = neighborhood_centroids[["LAT", "LON"]].values
    tree = cKDTree(centroid_coords)

    permit_coords = permits[["gis_latitude", "gis_longitude"]].values
    _, idx = tree.query(permit_coords, k=1)
    permits["NEIGHBORHOOD"] = neighborhood_centroids["NEIGHBORHOOD"].values[idx]

    permits["issuance_date"] = pd.to_datetime(permits["issuance_date"], errors="coerce")
    permits["PERMIT_YEAR"] = permits["issuance_date"].dt.year
    return permits.dropna(subset=["PERMIT_YEAR"])


def aggregate_permits_by_neighborhood_year(permits_with_neighborhood: pd.DataFrame) -> pd.DataFrame:
    agg = permits_with_neighborhood.groupby(["NEIGHBORHOOD", "PERMIT_YEAR"]).size().reset_index(name="PERMIT_COUNT")
    agg = agg.rename(columns={"PERMIT_YEAR": "BASE_YEAR"})
    agg["BASE_YEAR"] = agg["BASE_YEAR"].astype(int)
    return agg


def add_permit_feature(model_df: pd.DataFrame, permit_agg: pd.DataFrame) -> pd.DataFrame:
    """Merge permit counts onto model_df; neighborhoods/years with zero
    matched permits get PERMIT_COUNT=0 (a real zero, not missing data -
    it means no significant construction was issued that year)."""
    merged = model_df.merge(permit_agg, on=["NEIGHBORHOOD", "BASE_YEAR"], how="left")
    merged["PERMIT_COUNT"] = merged["PERMIT_COUNT"].fillna(0).astype(int)
    return merged


if __name__ == "__main__":
    # Requires df_spatial (or any geocoded sales df with NEIGHBORHOOD/LATITUDE/
    # LONGITUDE) and model_df to already exist from earlier pipeline steps.
    import visualization

    df_spatial = pd.read_csv(config.SALES_WITH_SPATIAL_CSV, low_memory=False)
    centroids = visualization.build_neighborhood_centroids(df_spatial)

    permits = fetch_manhattan_permits()
    permits_assigned = assign_permits_to_neighborhoods(permits, centroids)
    permit_agg = aggregate_permits_by_neighborhood_year(permits_assigned)
    print(permit_agg.sort_values("PERMIT_COUNT", ascending=False).head(10))

    model_df = pd.read_csv(config.GROWTH_MODEL_DATASET_CSV, low_memory=False)
    model_df = add_permit_feature(model_df, permit_agg)
    model_df.to_csv(config.GROWTH_MODEL_DATASET_CSV, index=False)
    print(f"\nSaved updated model dataset with PERMIT_COUNT -> {config.GROWTH_MODEL_DATASET_CSV}")
