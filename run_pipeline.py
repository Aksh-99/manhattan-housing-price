import os
 
import numpy as np
import pandas as pd
 
import config
import data_cleaning
import geocoding
import growth_target
import modeling
import spatial_features
import visualization
 
 
def main():
    # Clean sales data
    if os.path.exists(config.MODELING_READY_CSV):
        df = pd.read_csv(config.MODELING_READY_CSV, low_memory=False)
        print(f"Loaded cached cleaned data: {df.shape}")
    else:
        df = data_cleaning.load_and_clean()
        df.to_csv(config.MODELING_READY_CSV, index=False)
        print(f"Cleaned data: {df.shape}")
 
    visualization.plot_price_trend(df)
 
    # Geocoding
    if os.path.exists(config.GEOCODED_SALES_CSV):
        df_geocoded = pd.read_csv(config.GEOCODED_SALES_CSV, low_memory=False)
        print(f"Loaded cached geocoded data: {df_geocoded.shape}")
    else:
        unique_addrs = geocoding.prepare_unique_addresses(df)
        geocoded_addrs = geocoding.geocode_addresses(unique_addrs)
        geocoded_addrs.to_csv(config.GEOCODED_ADDRESSES_CSV, index=False)
        df_geocoded = geocoding.attach_coordinates(df, geocoded_addrs)
        df_geocoded.to_csv(config.GEOCODED_SALES_CSV, index=False)
 
    # Spatial features
    if os.path.exists(config.SALES_WITH_SPATIAL_CSV):
        df_spatial = pd.read_csv(config.SALES_WITH_SPATIAL_CSV, low_memory=False)
        amenities = pd.read_csv(config.AMENITIES_CSV)
        print(f"Loaded cached spatial features: {df_spatial.shape}")
    else:
        amenities = spatial_features.fetch_all_amenities()
        amenities.to_csv(config.AMENITIES_CSV, index=False)
        df_spatial = spatial_features.add_distance_features(df_geocoded, amenities)
        df_spatial.to_csv(config.SALES_WITH_SPATIAL_CSV, index=False)

    # Growth-rate target 
    neighborhood_year = growth_target.build_neighborhood_year_table(df)
    growth_df = growth_target.build_growth_target(neighborhood_year)
    growth_df = growth_target.add_price_momentum(growth_df, neighborhood_year)
    growth_df = growth_target.add_crisis_year_flag(growth_df)
    spatial_agg = growth_target.aggregate_spatial_features()

    raw_target_col = f"GROWTH_RATE_{config.FORWARD_YEARS}YR"
    log_target_col = f"LOG_{raw_target_col}"

    model_df = growth_df.merge(spatial_agg, on=["NEIGHBORHOOD", "BASE_YEAR"], how="left")

    centroids = visualization.build_neighborhood_centroids(df_spatial)

    mortgage_by_year = growth_target.fetch_mortgage_rate_by_year()
    model_df = model_df.merge(mortgage_by_year, on="BASE_YEAR", how="left")
 
    momentum_col = f"PRICE_MOMENTUM_{config.MOMENTUM_WINDOW_YEARS}YR"
    required_cols = [c for c in model_df.columns if c.startswith("AVG_")] + [momentum_col, "AVG_MORTGAGE_RATE"]
    model_df = model_df.dropna(subset=required_cols)
    model_df = growth_target.add_log_target(model_df, raw_target_col)
 
    model_df.to_csv(config.GROWTH_MODEL_DATASET_CSV, index=False)
    print(f"Model dataset: {model_df.shape}")
 
    # Modeling
    comparison = modeling.compare_random_vs_time_split(
        model_df, log_target_col, is_log_target=True, exclude_cols=[raw_target_col]
    )
    print("\nModel comparison:\n", comparison)

    print("\n=== ROLLING WINDOW (expanding-window, one test per year) ===")
    modeling.rolling_window_evaluate(
        model_df, log_target_col, is_log_target=True, exclude_cols=[raw_target_col]
    )

    model, X_train, X_test, y_train, y_test = modeling.train_primary_model(
        model_df, log_target_col, exclude_cols=[raw_target_col]
    )
 
    # Final gentrification-risk map
    latest_year = model_df["BASE_YEAR"].max()
    latest_data = model_df[model_df["BASE_YEAR"] == latest_year].copy()
    X_latest = latest_data.drop(columns=[log_target_col, raw_target_col, "NEIGHBORHOOD"])
 
    log_predictions = model.predict(X_latest)
    # Inverse-transform back to an interpretable growth-rate percentage
    latest_data["PREDICTED_GROWTH"] = np.sign(log_predictions) * np.expm1(np.abs(log_predictions))
 
    map_data = centroids.merge(
        latest_data[["NEIGHBORHOOD", "PREDICTED_GROWTH"]], on="NEIGHBORHOOD", how="inner"
    )
    visualization.build_gentrification_risk_map(map_data)
 
    print("\nPipeline complete.")
 
 
if __name__ == "__main__":
    main()
