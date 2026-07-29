import io
import time

import pandas as pd
import requests

import config


def prepare_unique_addresses(df: pd.DataFrame) -> pd.DataFrame:
    unique_addrs = (
        df[["ADDRESS", "ZIP_CODE"]]
        .dropna(subset=["ADDRESS"])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    unique_addrs["UNIQUE_ID"] = unique_addrs.index
    return unique_addrs


def _submit_chunk(chunk: pd.DataFrame, max_retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(max_retries):
        try:
            buf = io.StringIO()
            for _, row in chunk.iterrows():
                zip_code = str(int(row["ZIP_CODE"])) if pd.notna(row["ZIP_CODE"]) else ""
                addr = str(row["ADDRESS"]).replace(",", " ")
                buf.write(f"{row['UNIQUE_ID']},{addr},New York,NY,{zip_code}\n")

            files = {"addressFile": ("addresses.csv", buf.getvalue(), "text/csv")}
            data = {"benchmark": "Public_AR_Current"}

            resp = requests.post(config.CENSUS_BATCH_URL, files=files, data=data, timeout=240)
            resp.raise_for_status()

            return pd.read_csv(
                io.StringIO(resp.text),
                header=None,
                names=["UNIQUE_ID", "input_address", "match_status", "match_type",
                       "matched_address", "coordinates", "tiger_line_id", "side"],
            )
        except Exception as e:
            print(f"    attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def geocode_addresses(unique_addrs: pd.DataFrame) -> pd.DataFrame:
    all_results = []
    n_chunks = (len(unique_addrs) // config.CENSUS_CHUNK_SIZE) + 1

    for i in range(n_chunks):
        chunk = unique_addrs.iloc[i * config.CENSUS_CHUNK_SIZE:(i + 1) * config.CENSUS_CHUNK_SIZE]
        if len(chunk) == 0:
            continue
        print(f"Submitting chunk {i + 1}/{n_chunks} ({len(chunk)} addresses)...")
        result = _submit_chunk(chunk)
        if result is not None:
            all_results.append(result)
            print(f"  -> matched {(result['match_status'] == 'Match').sum()}/{len(result)}")
        else:
            print(f"  -> chunk {i + 1} FAILED after retries")
        time.sleep(2)

    geocoded = pd.concat(all_results, ignore_index=True)

    # A failed request that gets retried can occasionally produce a duplicate
    # UNIQUE_ID across chunks - keep the "Match" version when that happens.
    geocoded = geocoded.sort_values("match_status", ascending=False)
    geocoded = geocoded.drop_duplicates(subset="UNIQUE_ID", keep="first")

    matched = geocoded[geocoded["match_status"] == "Match"].copy()
    coords = matched["coordinates"].str.split(",", expand=True)
    matched["LONGITUDE"] = coords[0].astype(float)
    matched["LATITUDE"] = coords[1].astype(float)

    result = unique_addrs.merge(
        matched[["UNIQUE_ID", "LATITUDE", "LONGITUDE"]], on="UNIQUE_ID", how="left"
    )
    match_rate = result["LATITUDE"].notna().mean() * 100
    print(f"\nFinal match rate: {match_rate:.1f}% ({result['LATITUDE'].notna().sum()}/{len(result)})")
    return result


def attach_coordinates(df: pd.DataFrame, geocoded_addrs: pd.DataFrame) -> pd.DataFrame:
    """Merge lat/long back onto every sale row, drop rows that couldn't be geocoded."""
    merged = df.merge(
        geocoded_addrs[["ADDRESS", "ZIP_CODE", "LATITUDE", "LONGITUDE"]],
        on=["ADDRESS", "ZIP_CODE"], how="left",
    )
    coverage = merged["LATITUDE"].notna().mean() * 100
    print(f"Coordinate coverage: {coverage:.1f}% ({merged['LATITUDE'].notna().sum()}/{len(merged)})")
    return merged[merged["LATITUDE"].notna()].copy()


if __name__ == "__main__":
    from data_cleaning import load_and_clean

    df = load_and_clean()
    unique_addrs = prepare_unique_addresses(df)
    print("Unique addresses to geocode:", len(unique_addrs))

    geocoded_addrs = geocode_addresses(unique_addrs)
    geocoded_addrs.to_csv(config.GEOCODED_ADDRESSES_CSV, index=False)

    df_geocoded = attach_coordinates(df, geocoded_addrs)
    df_geocoded.to_csv(config.GEOCODED_SALES_CSV, index=False)
    print(f"Saved -> {config.GEOCODED_SALES_CSV} ({df_geocoded.shape})")
