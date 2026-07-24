# Manhattan Gentrification Risk Prediction

Predicts early-stage price growth by Manhattan neighborhood using 20 years
(2005-2025) of NYC DOF sales records, geospatial amenity data (transit,
parks, schools, groceries), and time-aware validation.

## Why this project

Most "housing price" portfolio projects predict price from a snapshot of
tabular features and evaluate with a random train/test split. This project
does two things differently:

1. **Predicts *future growth*, not current price** — the target is each
   neighborhood's price change 3 years forward, using only information
   available at the base year (a real early-warning framing, not a
   descriptive one).
2. **Validates with a time-based split** — training on early years and
   testing on later ones, rather than a random split. The gap between
   random-split and time-based-split R² is the core methodological
   finding: a naive random split meaningfully overstates real-world
   performance on this task (see `modeling.py` docstring for the numbers).

Framing note: the target is *price growth rate*, not "gentrification"
directly — price data alone can't support claims about displacement or
demographic change. Gentrification is the real-world motivation; growth-rate
prediction is the defensible, well-scoped ML problem underneath it.

## Pipeline

```
data_cleaning.py     -> clean the raw 20-year sales file
geocoding.py          -> geocode addresses (US Census Batch Geocoder)
spatial_features.py   -> pull transit/park/school/grocery data (Overpass API),
                          compute distance & count features
growth_target.py      -> build the neighborhood-year growth-rate target
modeling.py            -> compare Ridge / RandomForest / XGBoost, train primary model
visualization.py       -> price trend chart, interactive maps
run_pipeline.py         -> runs all of the above end-to-end
```

Run the full pipeline:

```bash
pip install -r requirements.txt
python run_pipeline.py
```

Note: `data_cleaning.py` expects a consolidated `manhattan_sales_2005_2025_clean.csv`
as input (20 years of raw DOF annual sales files merged into one schema —
see project notes for the consolidation script, not included here since
it's a one-time step). Geocoding and Overpass steps require internet access
and can take several minutes; results are cached to CSV so re-runs skip
straight to loading cached data.

## Key data-quality findings

A few things worth knowing if you're reading the code or reproducing this:

- **DOF files pad neighborhood/building-class names inconsistently across
  years** (e.g. `"SOHO"` vs `"SOHO                     "`, and internal
  double-spacing like `"13  CONDOS"` vs `"13 CONDOS"`). Left unfixed, this
  silently fragments any groupby into duplicate categories.
- **Co-op sales record `RESIDENTIAL_UNITS` as the whole building's unit
  count**, not the unit sold — a per-unit price feature computed naively
  from this field is meaningless for co-ops specifically.
- **Square footage is structurally missing** (not random) for ~90%+ of
  condo/co-op rows — DOF's sales file doesn't carry unit-level square
  footage for those transaction types. Imputing this would fabricate
  values for most of the dataset; it's flagged (`SQFT_KNOWN`) instead.
- **A neighborhood's price can spike hard in a single year purely because
  one new building's units all closed around the same time** (e.g. 775
  Riverside Drive in Washington Heights Lower drove an 8.8x apparent
  "growth" figure in 2023 that fully reverted the next year). Detected via
  a jump-then-reversion check on the year-over-year price series, and kept
  as a feature (`PRICE_SPIKE_YEAR`) rather than silently dropped, since
  it's a legitimate market event, just not the sustained trend the model
  is trying to predict.
- **Javits Center** was excluded entirely — its 2019 price series shows a
  permanent structural break coinciding with the Hudson Yards development
  opening nearby, not a repeatable growth pattern.

## Model results

RandomForest was the most robust of the three models tested on this
dataset (~700 neighborhood-year rows after feature engineering).
XGBoost consistently underperformed even after regularization tuning —
likely because its extra flexibility has little to work with at this
sample size. Full comparison numbers and SHAP feature-importance analysis
are in the accompanying notebook.
