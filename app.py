"""WeatherWise Flask application: pages + JSON API.

Start with:  python app.py   (then open http://127.0.0.1:5000)

The app never trains models. It loads the artifacts created by
`python -m ml.train_models` and the metrics JSON written by the same script.
"""
import json
import logging

from flask import Flask, jsonify, render_template, request

from ml import config
from ml.data import DatasetError, load_dataset, summarize_dataset
from ml.predict import (
    InputValidationError, ModelArtifactError, available_models,
    predict_cloud_cover, predict_rain,
)

app = Flask(__name__)
log = logging.getLogger("weatherwise")

_metrics_cache = {"mtime": None, "data": None}
_summary_cache = {"mtime": None, "data": None}


# ---------------------------------------------------------------------------
# Helpers that load saved results (re-read automatically after retraining)
# ---------------------------------------------------------------------------
def get_metrics():
    """Saved evaluation metrics, or None if the models have not been trained yet."""
    path = config.METRICS_PATH
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _metrics_cache["mtime"] != mtime:
        try:
            with open(path, encoding="utf-8") as fh:
                _metrics_cache["data"] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            log.exception("Could not read metrics file")
            return None
        _metrics_cache["mtime"] = mtime
    return _metrics_cache["data"]


def get_dataset_summary():
    """(summary, error_message) for the CSV currently in data/."""
    path = config.DATA_PATH
    try:
        mtime = path.stat().st_mtime if path.exists() else None
        if mtime is None or _summary_cache["mtime"] != mtime:
            _summary_cache["data"] = summarize_dataset(load_dataset(path))
            _summary_cache["mtime"] = mtime
        return _summary_cache["data"], None
    except DatasetError as exc:
        _summary_cache["mtime"] = None
        return None, str(exc)


def get_model_choices(path, task):
    try:
        return available_models(path, task), None
    except ModelArtifactError as exc:
        return None, str(exc)


def api_error(message, status, fields=None):
    body = {"ok": False, "error": message}
    if fields:
        body["fields"] = fields
    return jsonify(body), status


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    summary, data_error = get_dataset_summary()
    return render_template("index.html", active="home", summary=summary,
                           data_error=data_error, metrics=get_metrics())


@app.route("/rain-prediction")
def rain_prediction_page():
    choices, model_error = get_model_choices(config.RAIN_MODEL_PATH, "rain_classification")
    summary, _ = get_dataset_summary()
    return render_template("rain_prediction.html", active="rain", choices=choices,
                           model_error=model_error, summary=summary, metrics=get_metrics(),
                           features=config.RAIN_FEATURES, limits=config.HARD_LIMITS,
                           descriptions=config.FEATURE_DESCRIPTIONS)


@app.route("/cloud-regression")
def cloud_regression_page():
    choices, model_error = get_model_choices(config.CLOUD_MODEL_PATH, "cloud_regression")
    summary, _ = get_dataset_summary()
    return render_template("cloud_regression.html", active="cloud", choices=choices,
                           model_error=model_error, summary=summary, metrics=get_metrics(),
                           features=config.CLOUD_FEATURES, limits=config.HARD_LIMITS,
                           descriptions=config.FEATURE_DESCRIPTIONS)


@app.route("/model-evaluation")
def model_evaluation_page():
    summary, data_error = get_dataset_summary()
    return render_template("model_evaluation.html", active="eval", metrics=get_metrics(),
                           summary=summary, data_error=data_error)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
def _json_body():
    payload = request.get_json(silent=True)
    if payload is None:
        raise InputValidationError({"_": "Request body must be valid JSON."})
    return payload


@app.post("/api/predict-rain")
def api_predict_rain():
    try:
        result = predict_rain(_json_body())
    except InputValidationError as exc:
        return api_error("Please correct the highlighted inputs.", 400, exc.errors)
    except ModelArtifactError as exc:
        return api_error(str(exc), 503)
    return jsonify({"ok": True, **result})


@app.post("/api/predict-cloud-cover")
def api_predict_cloud_cover():
    try:
        result = predict_cloud_cover(_json_body())
    except InputValidationError as exc:
        return api_error("Please correct the highlighted inputs.", 400, exc.errors)
    except ModelArtifactError as exc:
        return api_error(str(exc), 503)

    # Attach the model's real held-out performance so the user can judge the estimate.
    metrics = get_metrics()
    try:
        test = metrics["regression"]["models"][result["model_key"]]["test"]
        result["test_r2"], result["test_rmse"] = test["r2"], test["rmse"]
    except (TypeError, KeyError):
        result["test_r2"] = result["test_rmse"] = None
    return jsonify({"ok": True, **result})


@app.get("/api/model-metrics")
def api_model_metrics():
    metrics = get_metrics()
    if metrics is None:
        return api_error("No evaluation metrics found. Run 'python -m ml.train_models' first.", 404)
    return jsonify({"ok": True, "metrics": metrics})


@app.get("/api/dataset-summary")
def api_dataset_summary():
    summary, error = get_dataset_summary()
    if error:
        return api_error(error, 503)
    return jsonify({"ok": True, "summary": summary})


# ---------------------------------------------------------------------------
# Error handling: JSON for the API, friendly pages otherwise, never stack traces
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    if request.path.startswith("/api/"):
        return api_error("Endpoint not found.", 404)
    return render_template("error.html", active=None, code=404,
                           message="That page does not exist."), 404


@app.errorhandler(405)
def method_not_allowed(_):
    if request.path.startswith("/api/"):
        return api_error("Method not allowed for this endpoint.", 405)
    return render_template("error.html", active=None, code=405,
                           message="Method not allowed."), 405


@app.errorhandler(Exception)
def unhandled(exc):
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return exc
    log.exception("Unhandled error")  # full details go to the server console only
    if request.path.startswith("/api/"):
        return api_error("Internal server error.", 500)
    return render_template("error.html", active=None, code=500,
                           message="Something went wrong on the server."), 500


@app.template_filter("pct")
def pct(value, digits=1):
    return "—" if value is None else f"{value * 100:.{digits}f}%"


@app.template_filter("num")
def num(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
