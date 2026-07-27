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
                          compute distance & count features (per-category cached
                          - a failed Overpass fetch for one amenity type doesn't
                          force re-fetching types that already succeeded)
growth_target.py      -> build the neighborhood-year growth-rate target, plus
                          PRICE_MOMENTUM_2YR, IS_CRISIS_YEAR, AVG_MORTGAGE_RATE
                          (FRED), and LOG_GROWTH_RATE_3YR (signed log1p transform)
modeling.py            -> compare Ridge / RandomForest / XGBoost; two validation
                          schemes (single time-based split, and rolling-window
                          walk-forward evaluation - see Model results below)
visualization.py       -> price trend chart + final gentrification-risk map
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
straight to loading cached data. Final output: `growth_model_dataset.csv`
(589 rows × 19 columns) and `manhattan_gentrification_risk_map.html`.

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

## Model results (final)

RandomForest, log-transformed target, full 16-feature set (neighborhood
identity, price/momentum, mortgage rate, crisis flag, all 8 spatial
distance/count columns):

| Validation scheme | R² |
|---|---|
| Random 80/20 split | 0.37 |
| Time-based split (train <2019, test ≥2019) | 0.10 |
| Rolling window (walk-forward, each year tested once) - mean | **-0.08** |
| Rolling window - crisis years only (2008-10, 2020-22) | -0.15 |
| Rolling window - non-crisis years only | -0.03 |

The random-vs-time-based gap (0.37 → 0.10) is the project's core
methodological finding: a naive random split substantially overstates
real-world performance, because it lets the model see information
adjacent in time to what it's tested on.

The rolling-window result is the more rigorous, and more honest, number.
Testing every year individually (rather than one blended post-2019 test
set) revealed the model is genuinely unstable year to year - not simply
"good except during crises." 2020 (the COVID crash itself) performed
reasonably (R²≈0.09); 2016, an ordinary non-crisis year, was one of the
worst performers in the whole series; 2021-2022 (the COVID rebound) is
where every configuration tested consistently fails hardest. This points
to a sample-size ceiling (~37 neighborhood-years per test year) more than
a fixable feature gap - no single feature addition closed this gap (see
below).

**What was tried, and explicitly rejected - don't re-add without a new
reason to revisit:**

- **NYC DOB building permits** (`permit_features.py` - kept in the repo as
  a standalone, documented script; not called by `run_pipeline.py`).
  Tested as a leading indicator (new construction / major alteration
  permits per neighborhood-year). No meaningful improvement (R² 0.093 vs.
  0.100 without). Likely redundant with price momentum - neighborhoods
  already showing rising prices plausibly also attract construction
  activity, so the signal was probably already captured indirectly.
- **Micro-zone granularity** (KMeans-clustered ~120 geographic zones
  instead of DOF's 43 neighborhoods) - file removed from the repo.
  Despite 3.6x more data points (2,121 vs. 589 rows), performed worse at
  every validation split, especially the rolling window (mean R² -0.37 vs.
  -0.08). Likely cause: purely geographic clustering doesn't respect real
  submarket boundaries (school zones, transit corridors, building stock)
  the way DOF's neighborhoods implicitly do, and at ~120 zones the average
  zone-year had only ~10-12 sales - too thin for a reliable median or
  2-year momentum calculation.
- **Trimming to a 6-feature "core set"** (dropping the 4 distance columns,
  weaker amenity counts, `PRICE_SPIKE_YEAR`, `SALES_COUNT`) based on SHAP
  individual-feature importance. This is the most interesting rejected
  idea: it looked like a win on the single time-based split (R² 0.10 →
  0.13), but the rolling window showed the opposite - mean R² collapsed
  from -0.08 to -0.37, with 2021 specifically cratering to R²=-2.2.
  Lesson: SHAP's per-feature importance doesn't capture *collective*
  stabilizing value - a group of individually-weak, correlated features
  (e.g. distance and count for the same amenity type) can still help a
  model generalize in atypical years, even when no single one of them
  looks important alone. The single time-based split missed this because
  it blends 2019-2022 into one number; the rolling window catches it
  because it never lets an unusually bad year hide inside an average.
- **`IS_CRISIS_YEAR` flag alone** - added hoping the model could learn "a
  crisis year behaves differently" as a general pattern. Barely moved the
  aggregate number, and the rolling window showed why: crisis years
  weren't uniquely bad (2016, non-crisis, was worse than most of them),
  so a binary crisis flag was the wrong lens for the actual problem
  (general year-to-year instability at this sample size).
- **Sales+amenities interactive heatmap** (`visualization.build_sales_amenities_map`)
  - removed from the pipeline; not part of the final deliverable.

Worth flagging honestly: an early version of the log-target comparison
showed a suspiciously perfect R² (~0.9997) on the random split. Cause:
the original un-logged growth-rate column was still sitting in the
feature set alongside its log-transformed version - the model was
effectively being handed the answer. Fixed via `exclude_cols` in
`modeling.compare_random_vs_time_split` / `train_primary_model`. Any
result that looks too good is worth this kind of suspicion before it
gets reported anywhere.

## What this model actually supports

Given the above, the honest framing is: **a ranking tool, not a
forecasting tool.** The data supports identifying neighborhoods that
share characteristics with historically fast-growing areas - low current
price and strong subway access were the two consistent, SHAP-confirmed
signals, and this pattern also emerged independently in raw historical
growth-rate rankings (Harlem repeatedly topped the list before any model
was involved). It does not support a precise, trustworthy growth-rate
prediction for any single neighborhood in a given year - the rolling-window
results show real, unresolved year-to-year instability that more feature
engineering did not fix.

