"""Loading, validating and summarising the weather CSV."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ml import config


class DatasetError(Exception):
    """Raised when the CSV is missing or does not match the expected schema."""


def load_dataset(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the CSV and validate it. Never generates replacement data.

    Checks performed:
      * the file exists and is a readable CSV
      * all six expected columns are present
      * feature columns are numeric and finite
      * the Rain column only contains the labels "rain" / "no rain"
      * rows with missing values are dropped (and counted in the summary)
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"Dataset not found. Place 'Weather_classification_Logistic.csv' in the "
            f"'data/' folder of the project (expected: {path.name})."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pandas raises several different error types
        raise DatasetError(f"Could not read the dataset as CSV: {exc}") from exc

    expected = config.ALL_FEATURES + [config.RAIN_TARGET]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise DatasetError(f"Dataset is missing required column(s): {', '.join(missing_cols)}")
    df = df[expected].copy()

    rows_read = len(df)
    missing_values = int(df.isna().sum().sum())
    df = df.dropna()

    # Feature columns must be numeric and finite.
    for col in config.ALL_FEATURES:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            raise DatasetError(f"Column '{col}' contains non-numeric values.")
        if not np.isfinite(converted).all():
            raise DatasetError(f"Column '{col}' contains infinite values.")
        df[col] = converted.astype(float)

    # Normalise label spelling/whitespace, then check against the known labels.
    df[config.RAIN_TARGET] = df[config.RAIN_TARGET].astype(str).str.strip().str.lower()
    bad_labels = sorted(set(df[config.RAIN_TARGET]) - set(config.RAIN_LABELS))
    if bad_labels:
        raise DatasetError(
            f"Unexpected value(s) in 'Rain': {bad_labels}. Allowed: {config.RAIN_LABELS}."
        )
    present = set(df[config.RAIN_TARGET])
    if present != set(config.RAIN_LABELS):
        raise DatasetError("The 'Rain' column must contain both 'rain' and 'no rain' examples.")
    if len(df) < 50:
        raise DatasetError(f"Dataset has only {len(df)} usable rows; at least 50 are needed.")

    df.attrs["rows_read"] = rows_read
    df.attrs["missing_values"] = missing_values
    df.attrs["rows_dropped"] = rows_read - len(df)
    return df.reset_index(drop=True)


def file_sha256(path: Path = config.DATA_PATH) -> str:
    """Fingerprint of the CSV, stored with the models to detect a changed dataset."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Row count, features, target distribution, missing values and feature stats."""
    counts = df[config.RAIN_TARGET].value_counts()
    total = int(len(df))
    distribution = {
        label: {"count": int(counts.get(label, 0)),
                "percent": round(100.0 * counts.get(label, 0) / total, 2)}
        for label in config.RAIN_LABELS
    }
    stats = {}
    for col in config.ALL_FEATURES:
        s = df[col]
        stats[col] = {
            "min": float(s.min()), "max": float(s.max()),
            "mean": float(s.mean()), "std": float(s.std()),
        }
    return {
        "rows": total,
        "rows_read": int(df.attrs.get("rows_read", total)),
        "rows_dropped": int(df.attrs.get("rows_dropped", 0)),
        "missing_values": int(df.attrs.get("missing_values", 0)),
        "features": list(config.ALL_FEATURES),
        "target": config.RAIN_TARGET,
        "target_distribution": distribution,
        "feature_stats": stats,
        "units_verified": False,
        "provenance_note": (
            "The CSV was supplied without documentation of its source or measurement "
            "units. Units are therefore not shown, and the data should not be assumed "
            "to represent real-world weather observations until verified."
        ),
    }
