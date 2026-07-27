"""
Visualizations for the Manhattan gentrification-risk project:
  1. Citywide median price trend (matplotlib, static)
  2. Interactive sales/amenities map (folium)
  3. Final gentrification-risk map - predicted growth rate by neighborhood,
     colored on a legend scale (folium + branca colormap)
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_price_trend(df: pd.DataFrame, save_path: str = "manhattan_price_trend.png"):
    yearly_median = df.groupby("SALE_YEAR")["SALE_PRICE"].median()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(yearly_median.index, yearly_median.values, marker="o", linewidth=2, color="#2563eb")
    ax.axvspan(2008, 2010, alpha=0.15, color="red", label="2008 Financial Crisis")
    ax.axvspan(2020, 2020.5, alpha=0.15, color="orange", label="COVID-19")

    ax.set_title("Manhattan Residential Median Sale Price by Year (2005-2025)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sale Year")
    ax.set_ylabel("Median Sale Price ($)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.set_xticks(range(2005, 2026, 2))
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved -> {save_path}")


def build_sales_amenities_map(df_props: pd.DataFrame, amenities: pd.DataFrame,
                                save_path: str = "manhattan_interactive_map.html"):
    """Heatmap of sale price density + toggleable amenity layers on a real
    OSM basemap. Uses a heatmap rather than individual markers for sales
    since plotting 250K+ raw points would freeze the browser."""
    import folium
    from folium.plugins import HeatMap, MarkerCluster

    center_lat, center_lon = df_props["LATITUDE"].median(), df_props["LONGITUDE"].median()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="cartodbpositron")

    sample = df_props.sample(min(50000, len(df_props)), random_state=42)
    heat_data = sample[["LATITUDE", "LONGITUDE", "SALE_PRICE"]].dropna().values.tolist()
    HeatMap(heat_data, radius=8, blur=6, max_zoom=13, name="Sale Price Heatmap").add_to(m)

    colors = {"subway": "red", "park": "green", "school": "blue", "grocery": "orange"}
    for label, color in colors.items():
        subset = amenities[amenities["TYPE"] == label]
        if len(subset) == 0:
            continue
        cluster = MarkerCluster(name=label.capitalize())
        for _, row in subset.iterrows():
            folium.CircleMarker(
                location=[row["LATITUDE"], row["LONGITUDE"]],
                radius=4, color=color, fill=True, fill_color=color,
                fill_opacity=0.7, popup=label,
            ).add_to(cluster)
        cluster.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(save_path)
    print(f"Saved -> {save_path}")


def build_gentrification_risk_map(map_data: pd.DataFrame, save_path: str = "manhattan_gentrification_risk_map.html"):
    """map_data must have columns: NEIGHBORHOOD, LAT, LON, PREDICTED_GROWTH."""
    import branca.colormap as cm
    import folium

    m = folium.Map(location=[40.78, -73.96], zoom_start=12, tiles="cartodbpositron")

    colormap = cm.LinearColormap(
        colors=["green", "yellow", "orange", "red"],
        vmin=map_data["PREDICTED_GROWTH"].min(),
        vmax=map_data["PREDICTED_GROWTH"].max(),
        caption="Predicted 3-Year Growth Rate (Gentrification Risk)",
    )

    for _, row in map_data.iterrows():
        color = colormap(row["PREDICTED_GROWTH"])
        folium.CircleMarker(
            location=[row["LAT"], row["LON"]],
            radius=14, color=color, fill=True, fill_color=color, fill_opacity=0.8,
            popup=f"{row['NEIGHBORHOOD']}: {row['PREDICTED_GROWTH']*100:.1f}% predicted growth",
            tooltip=row["NEIGHBORHOOD"],
        ).add_to(m)

    colormap.add_to(m)
    m.save(save_path)
    print(f"Saved -> {save_path}")


def build_neighborhood_centroids(df_props: pd.DataFrame) -> pd.DataFrame:
    return df_props.groupby("NEIGHBORHOOD").agg(
        LAT=("LATITUDE", "mean"), LON=("LONGITUDE", "mean")
    ).reset_index()
