"""Inference: load saved model bundles and make single predictions.

This module never trains anything. It only loads the artifacts written by
ml/train_models.py from the project's own models/ folder (fixed file names,
never a path supplied by a user), because joblib/pickle files can execute
code when loaded and must only come from a trusted source.
"""
import math
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from ml import config
from ml.evaluate import rain_decision, rain_probability


class ModelArtifactError(Exception):
    """Saved model is missing, unreadable or incompatible."""


class InputValidationError(Exception):
    """User input failed validation. `errors` maps field name -> message."""

    def __init__(self, errors: dict):
        super().__init__("Invalid input")
        self.errors = errors


RETRAIN_HINT = "Run 'python -m ml.train_models' from the project folder to (re)train the models."

_cache = {}  # path -> (modification time, bundle)


def load_bundle(path: Path, task: str) -> dict:
    """Load a model bundle, re-reading it automatically if the file was retrained."""
    path = Path(path)
    if not path.exists():
        raise ModelArtifactError(f"Trained model not found. {RETRAIN_HINT}")
    mtime = path.stat().st_mtime
    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelArtifactError(f"The saved model could not be loaded. {RETRAIN_HINT}") from exc

    expected_features = config.RAIN_FEATURES if task == "rain_classification" else config.CLOUD_FEATURES
    if not isinstance(bundle, dict) or bundle.get("format_version") != config.ARTIFACT_FORMAT_VERSION:
        raise ModelArtifactError(f"The saved model uses an outdated format. {RETRAIN_HINT}")
    if bundle.get("task") != task or bundle.get("features") != expected_features:
        raise ModelArtifactError(f"The saved model does not match this task. {RETRAIN_HINT}")
    if bundle.get("sklearn_version") != sklearn.__version__:
        raise ModelArtifactError(
            f"The saved model was trained with scikit-learn {bundle.get('sklearn_version')}, "
            f"but {sklearn.__version__} is installed. {RETRAIN_HINT}"
        )
    _cache[path] = (mtime, bundle)
    return bundle


def available_models(path: Path, task: str) -> dict:
    """{model_key: display name} plus the selected key, for the model dropdowns."""
    bundle = load_bundle(path, task)
    return {"selected": bundle["selected"],
            "models": {k: v["name"] for k, v in bundle["models"].items()}}


def _to_float(value):
    """Accept JSON numbers or numeric strings; reject booleans, blanks, NaN, inf."""
    if isinstance(value, bool) or value is None:
        raise ValueError("must be a number")
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("is required")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("must be a number") from None
    if not math.isfinite(number):
        raise ValueError("must be a finite number")
    return number


def validate_inputs(payload, features, training_ranges):
    """Check a JSON payload. Returns (values, warnings) or raises InputValidationError.

    * every feature must be present, numeric and finite
    * values outside HARD_LIMITS are rejected (impossible / obvious typo)
    * values outside the TRAINING range are accepted but produce a warning –
      they are never silently clipped
    * unexpected fields are rejected (e.g. Cloud_Cover sent to the regression API)
    """
    if not isinstance(payload, dict):
        raise InputValidationError({"_": "Request body must be a JSON object."})

    errors, values, warnings = {}, {}, []
    unexpected = sorted(set(payload) - set(features) - {"model"})
    for name in unexpected:
        errors[name] = "is not an input of this model"

    for name in features:
        if name not in payload:
            errors[name] = "is required"
            continue
        try:
            number = _to_float(payload[name])
        except ValueError as exc:
            errors[name] = str(exc)
            continue
        lo, hi = config.HARD_LIMITS[name]
        if not lo <= number <= hi:
            errors[name] = f"must be between {lo:g} and {hi:g}"
            continue
        values[name] = number
        tmin, tmax = training_ranges[name]
        if not tmin <= number <= tmax:
            warnings.append(
                f"{name} = {number:g} is outside the training range ({tmin:.2f} – {tmax:.2f}); "
                f"the model is extrapolating, so treat this prediction with extra caution."
            )

    if errors:
        raise InputValidationError(errors)
    return values, warnings


def _choose_model(bundle, payload):
    key = payload.get("model") or bundle["selected"]
    if key not in bundle["models"]:
        raise InputValidationError({"model": f"unknown model '{key}'"})
    return key, bundle["models"][key]


def predict_rain(payload) -> dict:
    """Rain class + estimated rain probability for one set of measurements."""
    bundle = load_bundle(config.RAIN_MODEL_PATH, "rain_classification")
    values, warnings = validate_inputs(payload, bundle["features"], bundle["training_ranges"])
    key, entry = _choose_model(bundle, payload)
    model = entry["estimator"]

    X = pd.DataFrame([values], columns=bundle["features"])
    probability = float(rain_probability(model, X)[0])
    threshold = bundle["threshold"]
    label = str(rain_decision([probability], threshold)[0])  # same rule used in evaluation

    return {
        "prediction": label,
        "display_label": "Rain" if label == config.POSITIVE_LABEL else "No Rain",
        "probability_rain": probability,
        "threshold": threshold,
        "model_key": key,
        "model_name": entry["name"],
        "interpretation": _rain_interpretation(probability, threshold, label),
        "warnings": warnings,
        "inputs": values,
    }


def _rain_interpretation(p, threshold, label):
    pct, thr = round(p * 100, 1), round(threshold * 100)
    side = "at or above" if label == config.POSITIVE_LABEL else "below"
    text = (f"The model's estimated rain probability of {pct}% is {side} the decision threshold "
            f"of {thr}%, so it predicts {'Rain' if label == config.POSITIVE_LABEL else 'No Rain'}.")
    margin = abs(p - threshold)
    if margin < 0.15:
        text += " The estimate is close to the threshold, so the model is fairly uncertain here."
    else:
        text += " The estimate is well away from the threshold, so the model is fairly confident."
    return text


def predict_cloud_cover(payload) -> dict:
    """Cloud-cover estimate from Temperature, Humidity, Wind_Speed and Pressure only."""
    bundle = load_bundle(config.CLOUD_MODEL_PATH, "cloud_regression")
    values, warnings = validate_inputs(payload, bundle["features"], bundle["training_ranges"])
    key, entry = _choose_model(bundle, payload)
    model = entry["estimator"]

    X = pd.DataFrame([values], columns=bundle["features"])
    raw = float(model.predict(X)[0])

    # Display policy: the target is on a 0–100 scale, so a value outside it is
    # not physically meaningful. We show the limited value AND the raw output,
    # and the evaluation metrics are always computed on raw outputs.
    lo, hi = config.CLOUD_COVER_RANGE
    shown = min(max(raw, lo), hi)
    clipped = shown != raw
    if clipped:
        warnings.append(
            f"The raw model output ({raw:.2f}) is outside the 0–100 scale of Cloud_Cover; "
            f"the displayed value has been limited to {shown:g}."
        )

    return {
        "prediction": shown,
        "raw_prediction": raw,
        "was_limited": clipped,
        "model_key": key,
        "model_name": entry["name"],
        "warnings": warnings,
        "inputs": values,
    }
