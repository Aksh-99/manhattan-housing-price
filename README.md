# Manhattan Gentrification Risk Ranking

Ranks Manhattan neighborhoods by how strongly they resemble historically
fast-growing areas — 20 years (2005-2025) of NYC DOF sales data, geospatial
amenity data, time-aware validation.

**Goal:** not a precise forecast for one neighborhood, but a defensible,
evidence-based watchlist. Price growth is the actual target (not
"gentrification" — price data alone can't support displacement claims).

## Run it

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Needs `manhattan_sales_2005_2025_clean.csv` as input (one-time
consolidation of 20 raw DOF files, not included). Geocoding/Overpass need
internet, take several minutes, cache to CSV. Output:
`growth_model_dataset.csv` + `manhattan_gentrification_risk_map.html`.

```
data_cleaning.py -> geocoding.py -> spatial_features.py -> growth_target.py
-> modeling.py -> visualization.py   (run_pipeline.py runs all of it)
```

## Model results

RandomForest, log target, 16 features:

| Validation | R² |
|---|---|
| Random 80/20 split | 0.37 |
| Time-based split (train <2019, test ≥2019) | 0.10 |
| Rolling window, mean (each year tested once) | **-0.08** |

The random-vs-time gap is the core finding: naive random splits
overstate real performance. The rolling window is more honest and
messier — 2020 (COVID crash) did fine, 2016 (no crisis) was one of the
worst years, 2021-22 (rebound) fails hardest. Reads as a sample-size
ceiling (~37 rows/test-year), not a fixable feature gap.

## Data-quality gotchas worth knowing

- Neighborhood/building-class names have inconsistent whitespace across
  years — silently fragments any groupby if unfixed.
- Co-op `RESIDENTIAL_UNITS` = whole building, not the unit sold.
- Square footage missing for ~90% of condo/co-op rows (not random —
  DOF doesn't track it for those types). Flagged, not imputed.
- A single building's launch can fake a neighborhood-wide price spike
  (e.g. 775 Riverside Drive, 2023 — 8.8x jump, fully reverted next year).
- Javits Center excluded — permanent break from Hudson Yards opening,
  not a repeatable pattern.

## Tried and rejected

- **Building permits** — no improvement, redundant with price momentum.
- **Micro-zones** (KMeans, ~120 zones) — worse at every split despite
  3.6x more rows; too few sales/zone-year for reliable momentum.
- **6-feature "core set"** (SHAP-based trim) — looked better on one
  split, collapsed on the rolling window (misses features' *collective*
  stabilizing value).
- **Crisis-year flag alone** — barely helped; non-crisis years were
  often just as bad.

Also caught: an early run showed a suspicious ~1.0 R² — the un-logged
target column was leaking into the features. Fixed via `exclude_cols`
in `modeling.py`. Any too-good result deserves that suspicion.

## Bottom line

Low price + strong subway access reliably predicted growth — confirmed
by SHAP and by Harlem independently topping raw rankings pre-model.
That's the ranking claim, and it held up. It does **not** give a
trustworthy precise forecast for one neighborhood in one year — the
rolling window shows real year-to-year instability nothing here fixed.

