"""Central configuration shared by training, inference, the Flask app and tests.

Keeping every path, column name and constant in one place means the training
script and the web app can never disagree about (for example) which columns
are features.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Weather_classification_Logistic.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RAIN_MODEL_PATH = MODELS_DIR / "rain_classifier.joblib"
CLOUD_MODEL_PATH = MODELS_DIR / "cloud_regressor.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"

# Bumped whenever the structure of the saved artifact bundles changes, so old
# artifacts are detected as incompatible instead of failing in odd ways.
ARTIFACT_FORMAT_VERSION = 1

# ---------------------------------------------------------------------------
# Dataset schema (exact column names and labels from the supplied CSV)
# ---------------------------------------------------------------------------
ALL_FEATURES = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]
RAIN_TARGET = "Rain"
RAIN_LABELS = ["no rain", "rain"]
POSITIVE_LABEL = "rain"  # the minority class we care most about

# Task 1 – rain classification uses all five measurements.
RAIN_FEATURES = ["Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Pressure"]

# Task 2 – the professor's multiple regression:
#   Y = Cloud_Cover,  X = Temperature, Humidity, Wind_Speed, Pressure
# Cloud_Cover is the target, so it must NEVER appear in the inputs.
CLOUD_TARGET = "Cloud_Cover"
CLOUD_FEATURES = ["Temperature", "Humidity", "Wind_Speed", "Pressure"]

# ---------------------------------------------------------------------------
# Experimental setup
# ---------------------------------------------------------------------------
RANDOM_STATE = 42     # fixed seed => identical splits and models on every run
TEST_SIZE = 0.20      # 80:20 train/test split
CV_FOLDS = 5          # k-fold cross-validation on the TRAINING portion only

# Decision threshold for the rain class. We keep the conventional 0.5: the
# classifiers already use class_weight="balanced" to counter the imbalance,
# so no threshold was tuned (and certainly not on the test set).
RAIN_DECISION_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Input validation for the prediction forms / API
# ---------------------------------------------------------------------------
# The CSV ships without documentation, so measurement units are NOT known and
# are never displayed. These "hard" limits only reject values that are
# impossible for the quantity whatever its unit (negative wind speed, humidity
# above 100 on a 0-100 scale, ...) or that are obviously typos. Values that are
# valid but outside the range seen in training are accepted and flagged with
# a warning (extrapolation) – they are never silently clipped.
HARD_LIMITS = {
    "Temperature": (-100.0, 150.0),   # loose sanity bound, unit unverified
    "Humidity": (0.0, 100.0),         # column is on a 0-100 scale
    "Wind_Speed": (0.0, 500.0),       # a speed cannot be negative
    "Cloud_Cover": (0.0, 100.0),      # column is on a 0-100 scale
    "Pressure": (1.0, 2000.0),        # must be positive; upper bound catches typos
}

FEATURE_DESCRIPTIONS = {
    "Temperature": "Air temperature reading",
    "Humidity": "Relative humidity reading (0–100 scale in the dataset)",
    "Wind_Speed": "Wind speed reading",
    "Cloud_Cover": "Cloud cover reading (0–100 scale in the dataset)",
    "Pressure": "Atmospheric pressure reading",
}

# Physically meaningful range for cloud cover predictions (same scale as the
# target column). Used only for DISPLAY – see ml/predict.py.
CLOUD_COVER_RANGE = (0.0, 100.0)
