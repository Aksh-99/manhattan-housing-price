# Manhattan Gentrification Risk Ranking

Ranks Manhattan neighborhoods by how strongly they resemble historically
fast-growing areas — 20 years (2005-2025) of NYC DOF sales data, geospatial
amenity data, time-aware validation.

📝 [Full writeup on Medium](https://medium.com/@gopakumarakshara1999/predicting-gentrification-crystal-ball-or-rearview-mirror-8a7d63f022db)

**Goal:** not a precise forecast for one neighborhood, but a defensible,
evidence-based watchlist. Price growth is the target — not "gentrification"
itself, since price data alone can't support displacement claims.

## Model results

RandomForest, log target, 16 features:

| Validation | R² |
|---|---|
| Random split | 0.37 |
| Time-based split | 0.10 |
| Rolling window (each year tested once) | **-0.08** |

Random vs. time-based is the core finding: naive splits overstate real
performance. The rolling window is more honest and messier — 2020 (COVID)
did fine, 2016 (no crisis) was one of the worst years. Dug into 2016
specifically (worst miss: Chinatown, predicted +50%, actual -35%; 7 other
neighborhoods missed too, no common cause) — confirms a sample-size
ceiling, not a fixable bug.

## Data-quality notes

- Neighborhood/building-class names have inconsistent whitespace across
  years — fragments any groupby if unfixed.
- Co-op `RESIDENTIAL_UNITS` = whole building, not the unit sold.
- ~90% of condo/co-op square footage is genuinely untracked by DOF —
  flagged, not imputed.
- One building's launch can fake a neighborhood price spike (e.g. 775
  Riverside Drive, 2023 — 8.8x jump, fully reverted next year).
- Javits Center excluded — permanent Hudson Yards break, not a pattern.

## Tried and rejected

- Building permits — redundant with price momentum.
- Micro-zones (~120 KMeans clusters) — worse despite 3.6x more rows.
- 6-feature "core set" (SHAP-trimmed) — fine on one split, collapsed on
  rolling window.
- Crisis-year flag alone — barely helped.
- Looser spike thresholds — caught one real case, added noise elsewhere,
  R² unchanged. Reverted.
- Caught a leaked-target bug early on (~1.0 R² was too good to be true).

## Bottom line

Low price + subway access reliably predicted growth — confirmed by SHAP
and by Harlem topping raw rankings independently. The ranking claim held.
It does **not** give a trustworthy forecast for one neighborhood in one
year, and checking the worst year confirmed that's not a fixable bug.

