# File paths
RAW_SALES_CSV = "manhattan_sales_2005_2025_clean.csv"         
MODELING_READY_CSV = "manhattan_sales_2005_2025_modeling_ready.csv"
GEOCODED_ADDRESSES_CSV = "manhattan_addresses_geocoded.csv"
GEOCODED_SALES_CSV = "manhattan_sales_geocoded.csv"
AMENITIES_CSV = "manhattan_amenities.csv"
SALES_WITH_SPATIAL_CSV = "manhattan_sales_with_spatial_features.csv"
GROWTH_MODEL_DATASET_CSV = "growth_model_dataset.csv"
SPATIAL_AGG_CSV = "neighborhood_year_spatial_features.csv"
 
# Cleaning thresholds
MIN_SALE_PRICE = 10_000          
UPPER_PRICE_PERCENTILE = 0.995   
MIN_SALES_PER_NEIGHBORHOOD_YEAR = 10 
 
RESIDENTIAL_BUILDING_CLASSES = [
    "01 ONE FAMILY DWELLINGS",
    "02 TWO FAMILY DWELLINGS",
    "03 THREE FAMILY DWELLINGS",
    "09 COOPS - WALKUP APARTMENTS",
    "10 COOPS - ELEVATOR APARTMENTS",
    "12 CONDOS - WALKUP APARTMENTS",
    "13 CONDOS - ELEVATOR APARTMENTS",
]
 

EXCLUDED_NEIGHBORHOODS = ["MANHATTAN-UNKNOWN", "UPPER BAY", "JAVITS CENTER"]
 
# Growth-rate target
FORWARD_YEARS = 3          
PRICE_SPIKE_JUMP_THRESHOLD = 0.5    
PRICE_SPIKE_REVERSION_THRESHOLD = -0.3  
TIME_SPLIT_YEAR = 2019     
MOMENTUM_WINDOW_YEARS = 2  
 

CRISIS_YEARS = [2008, 2009, 2010, 2020, 2021, 2022]
 
# External macro data
FRED_MORTGAGE_RATE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
 
# Geocoding 
CENSUS_BATCH_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
CENSUS_CHUNK_SIZE = 1000
 
# Building permits 
DOB_PERMITS_URL = "https://data.cityofnewyork.us/resource/ipu4-2q9a.json"
DOB_PERMITS_CACHE = "manhattan_dob_permits.csv"
DOB_PERMIT_CHUNK_SIZE = 50_000

SIGNIFICANT_JOB_TYPES = ["NB", "A1"]
 
# Spatial features
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_USER_AGENT = "ManhattanHousingProject/1.0 (student research project)"
MANHATTAN_BBOX = "40.680,-74.020,40.880,-73.907"  # south, west, north, east
 

AMENITY_CACHE_TEMPLATE = "manhattan_amenities_{label}.csv"
 
