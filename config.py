
# --- File paths (raw input / intermediate / final outputs) ---
RAW_SALES_CSV = "manhattan_sales_2005_2025_clean.csv"          # output of consolidation step
MODELING_READY_CSV = "manhattan_sales_2005_2025_modeling_ready.csv"
GEOCODED_ADDRESSES_CSV = "manhattan_addresses_geocoded.csv"
GEOCODED_SALES_CSV = "manhattan_sales_geocoded.csv"
AMENITIES_CSV = "manhattan_amenities.csv"
SALES_WITH_SPATIAL_CSV = "manhattan_sales_with_spatial_features.csv"
GROWTH_MODEL_DATASET_CSV = "growth_model_dataset.csv"
SPATIAL_AGG_CSV = "neighborhood_year_spatial_features.csv"
 
# --- Cleaning thresholds ---
MIN_SALE_PRICE = 10_000          # drop $0 / nominal non-market transfers
UPPER_PRICE_PERCENTILE = 0.995   # cap ultra-luxury outlier tail
MIN_SALES_PER_NEIGHBORHOOD_YEAR = 10  # drop sparse neighborhood-year cells
 
RESIDENTIAL_BUILDING_CLASSES = [
    "01 ONE FAMILY DWELLINGS",
    "02 TWO FAMILY DWELLINGS",
    "03 THREE FAMILY DWELLINGS",
    "09 COOPS - WALKUP APARTMENTS",
    "10 COOPS - ELEVATOR APARTMENTS",
    "12 CONDOS - WALKUP APARTMENTS",
    "13 CONDOS - ELEVATOR APARTMENTS",
]
 
# Neighborhoods excluded for data-quality / non-residential reasons (see README
# for the investigation behind each): DOF catch-all with no post-2016 data,
# a 2-row non-market area (Governors Island), and a structural break caused
# by the Hudson Yards mega-development launch (not organic price growth).
EXCLUDED_NEIGHBORHOODS = ["MANHATTAN-UNKNOWN", "UPPER BAY", "JAVITS CENTER"]
 
# --- Growth-rate target ---
FORWARD_YEARS = 3          # predict growth over this many years ahead
PRICE_SPIKE_JUMP_THRESHOLD = 0.5     # >50% YoY jump
PRICE_SPIKE_REVERSION_THRESHOLD = -0.3  # followed by >30% drop the next year
TIME_SPLIT_YEAR = 2019     # train on BASE_YEAR < this, test on >= this
MOMENTUM_WINDOW_YEARS = 2  # trailing window for the price-momentum feature
 
# Years marking known market-wide crisis/disruption periods, independent of
# any single neighborhood's price behavior - the 2008 financial crisis and
# the COVID crash/rebound. Used as an IS_CRISIS_YEAR feature so the model
# has at least two historical examples of "a crisis year behaves
# differently" rather than being blindsided by COVID as a total unknown.
CRISIS_YEARS = [2008, 2009, 2010, 2020, 2021, 2022]
 
# --- External macro data (FRED, free, no API key) ---
FRED_MORTGAGE_RATE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
 
# --- Geocoding (US Census Batch Geocoder) ---
CENSUS_BATCH_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
CENSUS_CHUNK_SIZE = 1000
 
# --- Building permits (NYC DOB Permit Issuance, free, no API key) ---
DOB_PERMITS_URL = "https://data.cityofnewyork.us/resource/ipu4-2q9a.json"
DOB_PERMITS_CACHE = "manhattan_dob_permits.csv"
DOB_PERMIT_CHUNK_SIZE = 50_000
# NB = New Building, A1 = major Alteration (the two job types that represent
# genuinely significant construction activity - a real leading indicator -
# as opposed to A2/A3 which cover minor work like a bathroom renovation)
SIGNIFICANT_JOB_TYPES = ["NB", "A1"]
 
# --- Spatial features (Overpass API / OpenStreetMap) ---
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_USER_AGENT = "ManhattanHousingProject/1.0 (student research project)"
MANHATTAN_BBOX = "40.680,-74.020,40.880,-73.907"  # south, west, north, east
 
# Per-category amenity cache filenames (see spatial_features.fetch_all_amenities) -
# Overpass rate-limits aggressively and a single category failing shouldn't
# force re-fetching categories that already succeeded.
AMENITY_CACHE_TEMPLATE = "manhattan_amenities_{label}.csv"
 