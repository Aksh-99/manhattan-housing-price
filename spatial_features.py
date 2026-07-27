import time
 
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
 
import config
 
QUERIES = {
    "subway": """
        [out:json][timeout:180];
        (
          node["railway"="subway_entrance"]({bbox});
          node["station"="subway"]({bbox});
        );
        out center;
    """,
    "park": """
        [out:json][timeout:180];
        (
          way["leisure"="park"]({bbox});
          relation["leisure"="park"]({bbox});
        );
        out center;
    """,
    "school": """
        [out:json][timeout:180];
        (
          node["amenity"="school"]({bbox});
          way["amenity"="school"]({bbox});
        );
        out center;
    """,
    "grocery": """
        [out:json][timeout:180];
        (
          node["shop"="supermarket"]({bbox});
          node["shop"="grocery"]({bbox});
        );
        out center;
    """,
}
 
 
def query_overpass(query: str, max_retries: int = 3) -> dict | None:
    headers = {"User-Agent": config.OVERPASS_USER_AGENT}
    for attempt in range(max_retries):
        try:
            resp = requests.post(config.OVERPASS_URL, data={"data": query}, headers=headers, timeout=180)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(15 * (attempt + 1))  # Overpass rate-limits hard; back off generously
    return None
 
 
def _elements_to_df(osm_json: dict | None, label: str) -> pd.DataFrame:
    if osm_json is None:
        # Explicitly typed, even when empty - an untyped/object-dtype empty
        # frame silently upcasts LATITUDE/LONGITUDE to object across ALL
        # amenity types once concatenated in fetch_all_amenities, even ones
        # that queried successfully. That upcast then breaks np.radians()
        # deep in haversine_km with a confusing "float has no attribute
        # radians" error far from the actual cause.
        return pd.DataFrame({
            "LATITUDE": pd.Series(dtype="float64"),
            "LONGITUDE": pd.Series(dtype="float64"),
            "TYPE": pd.Series(dtype="object"),
        })
    rows = []
    for el in osm_json.get("elements", []):
        if el["type"] == "node":
            lat, lon = el.get("lat"), el.get("lon")
        else:
            center = el.get("center", {})
            lat, lon = center.get("lat"), center.get("lon")
        if lat is not None and lon is not None:
            rows.append({"LATITUDE": lat, "LONGITUDE": lon, "TYPE": label})
    if not rows:
        return _elements_to_df(None, label)  # same empty-but-typed fallback
    return pd.DataFrame(rows)
 
 
def fetch_all_amenities(force_refresh: bool = False) -> pd.DataFrame:
    """Fetches each amenity category separately, caching each to its own CSV
    as soon as it succeeds. If a later category fails (Overpass rate-limits
    hard - see retry/backoff in query_overpass), previously-successful
    categories are loaded from cache instead of re-queried. Pass
    force_refresh=True to ignore existing per-category caches and re-fetch
    everything."""
    import os
 
    frames = []
    for label, query_template in QUERIES.items():
        cache_path = config.AMENITY_CACHE_TEMPLATE.format(label=label)
 
        if not force_refresh and os.path.exists(cache_path):
            amenity_df = pd.read_csv(cache_path)
            print(f"Loaded cached {label}: {len(amenity_df)} locations")
            frames.append(amenity_df)
            continue
 
        print(f"Querying Overpass for: {label}...")
        query = query_template.format(bbox=config.MANHATTAN_BBOX)
        result = query_overpass(query)
        amenity_df = _elements_to_df(result, label)
        print(f"  -> found {len(amenity_df)} {label} locations")
 
        if len(amenity_df) > 0:
            amenity_df.to_csv(cache_path, index=False)
        else:
            print(f"  WARNING: 0 {label} locations - not caching, will retry on next run")
 
        frames.append(amenity_df)
        time.sleep(5)  # space out requests to avoid rate limiting
 
    return pd.concat(frames, ignore_index=True)
 
 
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))
 
 
def add_distance_features(df_props: pd.DataFrame, amenities: pd.DataFrame) -> pd.DataFrame:
    """For each property, add nearest-distance (km) and amenity counts
    within 500m/1km, for every amenity type in QUERIES. Iterates over the
    known QUERIES keys (not just amenities["TYPE"].unique()) so that a
    failed Overpass query for one category still produces NaN/0 columns
    for that type, rather than silently omitting it - otherwise the
    output schema would vary run to run depending on which of Overpass's
    queries happened to succeed."""
    df_props = df_props.copy()
    prop_coords = df_props[["LATITUDE", "LONGITUDE"]].values
 
    for label in QUERIES.keys():
        subset = amenities[amenities["TYPE"] == label]
 
        if len(subset) == 0:
            print(f"  WARNING: 0 '{label}' locations available - filling NaN/0 for this type")
            df_props[f"DIST_NEAREST_{label.upper()}_KM"] = np.nan
            df_props[f"COUNT_{label.upper()}_500M"] = 0
            df_props[f"COUNT_{label.upper()}_1KM"] = 0
            continue
 
        amenity_coords = subset[["LATITUDE", "LONGITUDE"]].values.astype(float)
        tree = cKDTree(amenity_coords)
 
        dist_deg, idx = tree.query(prop_coords, k=1)
        nearest = amenity_coords[idx]
        df_props[f"DIST_NEAREST_{label.upper()}_KM"] = haversine_km(
            prop_coords[:, 0], prop_coords[:, 1], nearest[:, 0], nearest[:, 1]
        )
 
        for radius_km, radius_deg, colname in [
            (0.5, 0.008, f"COUNT_{label.upper()}_500M"),
            (1.0, 0.015, f"COUNT_{label.upper()}_1KM"),
        ]:
            counts = []
            for lat, lon in prop_coords:
                cand_idx = tree.query_ball_point([lat, lon], r=radius_deg)
                if not cand_idx:
                    counts.append(0)
                    continue
                cand = amenity_coords[cand_idx]
                d = haversine_km(lat, lon, cand[:, 0], cand[:, 1])
                counts.append(int((d <= radius_km).sum()))
            df_props[colname] = counts
 
    return df_props
 
 
if __name__ == "__main__":
    df_props = pd.read_csv(config.GEOCODED_SALES_CSV, low_memory=False)
    print("Properties to process:", len(df_props))
 
    amenities = fetch_all_amenities()
    amenities.to_csv(config.AMENITIES_CSV, index=False)
    print(f"Saved {len(amenities)} amenity locations -> {config.AMENITIES_CSV}")
 
    df_props = add_distance_features(df_props, amenities)
    df_props.to_csv(config.SALES_WITH_SPATIAL_CSV, index=False)
    print(f"Saved -> {config.SALES_WITH_SPATIAL_CSV} ({df_props.shape})")