"""Train, evaluate and save every WeatherWise model.

Run from the project root:

    python -m ml.train_models

What happens, step by step:
  1. Load and validate data/Weather_classification_Logistic.csv.
  2. TASK 1 – rain classification (target: Rain, features: all five columns)
       * stratified 80:20 train/test split (seed 42)
       * Logistic Regression (scaled) vs Random Forest, both class-weighted
       * hyper-parameters + model choice decided with 5-fold CV on TRAINING data
  3. TASK 2 – cloud-cover multiple regression
       (target: Cloud_Cover, features: Temperature, Humidity, Wind_Speed, Pressure)
       * its own 80:20 split (seed 42), identical for both regressors
       * Multiple Linear Regression vs Random Forest Regressor
       * model choice decided with 5-fold CV on TRAINING data
  4. Evaluate every model ONCE on its untouched test set.
  5. Save model bundles (joblib) and all metrics (JSON) into models/.
"""
import json
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from ml import config
from ml.data import DatasetError, file_sha256, load_dataset, summarize_dataset
from ml.evaluate import classification_metrics, regression_metrics

POS = config.POSITIVE_LABEL


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _training_ranges(X) -> dict:
    """Min/max of each feature in the TRAINING data (used for extrapolation warnings)."""
    return {c: [float(X[c].min()), float(X[c].max())] for c in X.columns}


def _cv_summary(search: GridSearchCV, metric_names) -> dict:
    """Mean / std / standard error of the CV scores for the chosen hyper-parameters."""
    i = search.best_index_
    k = search.n_splits_
    out = {}
    for name in metric_names:
        mean = float(search.cv_results_[f"mean_test_{name}"][i])
        std = float(search.cv_results_[f"std_test_{name}"][i])
        if name.startswith("neg_"):  # sklearn maximises, so error metrics are negated
            name, mean = name[4:], -mean
        out[f"{name}_mean"] = mean
        out[f"{name}_std"] = std
        out[f"{name}_se"] = std / np.sqrt(k)
    return out


def _one_se_choice(models: dict, score_key: str, higher_is_better: bool, simple_key: str):
    """Pick the best model by CV score, using the one-standard-error rule.

    If the simpler model's CV score is within one standard error of the best
    model's score, the simpler model is preferred (the difference is within
    the noise of cross-validation, so the extra complexity is not justified).
    """
    sign = 1 if higher_is_better else -1
    best_key = max(models, key=lambda k: sign * models[k]["cv"][f"{score_key}_mean"])
    best = models[best_key]["cv"]
    simple = models[simple_key]["cv"]
    gap = sign * (best[f"{score_key}_mean"] - simple[f"{score_key}_mean"])
    if best_key != simple_key and gap <= best[f"{score_key}_se"]:
        return simple_key, (
            f"{models[best_key]['name']} scored slightly better in cross-validation, but the "
            f"difference ({gap:.4f}) is within one standard error ({best[f'{score_key}_se']:.4f}), "
            f"so the simpler {models[simple_key]['name']} is preferred."
        )
    if best_key == simple_key:
        return best_key, f"{models[best_key]['name']} had the best mean cross-validated {score_key.upper()}."
    return best_key, (
        f"{models[best_key]['name']} beat {models[simple_key]['name']} by {gap:.4f} in mean "
        f"cross-validated {score_key.upper()}, more than one standard error "
        f"({best[f'{score_key}_se']:.4f}), so the extra complexity is justified."
    )


# ---------------------------------------------------------------------------
# Task 1: rain classification
# ---------------------------------------------------------------------------
def train_rain_classifiers(df):
    X = df[config.RAIN_FEATURES]
    y = df[config.RAIN_TARGET]

    # Stratify so the ~12.6% rain share is the same in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {
        "f1": make_scorer(f1_score, pos_label=POS),
        "recall": make_scorer(recall_score, pos_label=POS),
        "precision": make_scorer(precision_score, pos_label=POS, zero_division=0),
        "roc_auc": "roc_auc",  # positive class = classes_[1] = "rain"
    }

    candidates = {
        "logistic_regression": {
            "name": "Logistic Regression",
            "description": (
                "Linear model of the log-odds of rain. Features are standardised inside the "
                "pipeline (scaler fitted on training folds only). class_weight='balanced' "
                "gives the rare 'rain' class more weight during training."
            ),
            # Scaling lives INSIDE the pipeline, so in every CV fold the scaler is
            # fitted on that fold's training part only => no leakage.
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000,
                                             random_state=config.RANDOM_STATE)),
            ]),
            "grid": {"model__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
        "random_forest": {
            "name": "Random Forest Classifier",
            "description": (
                "Ensemble of decision trees trained on bootstrap samples; can learn non-linear "
                "threshold rules and feature interactions. Trees do not need feature scaling. "
                "class_weight='balanced' counters the class imbalance."
            ),
            "estimator": RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=config.RANDOM_STATE
            ),
            "grid": {"max_depth": [None, 8], "min_samples_leaf": [1, 5]},
        },
    }

    results, fitted = {}, {}
    for key, spec in candidates.items():
        print(f"  [rain] tuning {spec['name']} with {config.CV_FOLDS}-fold CV ...")
        search = GridSearchCV(spec["estimator"], spec["grid"], scoring=scoring, refit="f1",
                              cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        fitted[key] = model

        test = classification_metrics(model, X_test, y_test)
        train = classification_metrics(model, X_train, y_train)
        results[key] = {
            "name": spec["name"],
            "description": spec["description"],
            "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "cv": _cv_summary(search, ["f1", "recall", "precision", "roc_auc"]),
            "train": {"accuracy": train["accuracy"], "f1_rain": train["f1_rain"],
                      "recall_rain": train["recall_rain"]},
            "test": test,
            "interpretation": _rain_model_interpretation(key, model),
        }

    selected, reason = _one_se_choice(results, "f1", True, simple_key="logistic_regression")

    # Reference point: always predicting the majority class "no rain".
    majority_acc = float((y_test != POS).mean())

    # Dataset audit: how well can a tiny, human-readable decision tree (depth 3)
    # explain the label? Fitted on training data, scored on test data.
    tree = DecisionTreeClassifier(max_depth=3, random_state=config.RANDOM_STATE).fit(X_train, y_train)
    audit = {
        "tree_max_depth": 3,
        "tree_test_accuracy": float((tree.predict(X_test) == y_test).mean()),
        "tree_rules": export_text(tree, feature_names=list(X.columns), decimals=1),
    }

    summary = {
        "features": config.RAIN_FEATURES,
        "target": config.RAIN_TARGET,
        "positive_label": POS,
        "threshold": config.RAIN_DECISION_THRESHOLD,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_rain_share": float((y_train == POS).mean()),
        "test_rain_share": float((y_test == POS).mean()),
        "cv_folds": config.CV_FOLDS,
        "selection_metric": "Mean 5-fold cross-validated F1-score of the 'rain' class (training data only)",
        "selection_rule": (
            "Highest mean CV F1 for the 'rain' class; if the simpler Logistic Regression is within "
            "one standard error of the best, prefer it. The test set is NOT used for this choice."
        ),
        "selected": selected,
        "selection_reason": reason,
        "majority_baseline_accuracy": majority_acc,
        "audit": audit,
        "models": results,
    }
    bundle = {
        "format_version": config.ARTIFACT_FORMAT_VERSION,
        "task": "rain_classification",
        "sklearn_version": sklearn.__version__,
        "trained_at": _now(),
        "features": config.RAIN_FEATURES,
        "target": config.RAIN_TARGET,
        "positive_label": POS,
        "threshold": config.RAIN_DECISION_THRESHOLD,
        "training_ranges": _training_ranges(X_train),
        "selected": selected,
        "models": {k: {"name": results[k]["name"], "estimator": fitted[k]} for k in fitted},
    }
    return bundle, summary


def _rain_model_interpretation(key, model):
    if key == "logistic_regression":
        coefs = model.named_steps["model"].coef_[0]
        return {"type": "standardised_coefficients",
                "values": dict(zip(config.RAIN_FEATURES, map(float, coefs)))}
    return {"type": "feature_importances",
            "values": dict(zip(config.RAIN_FEATURES, map(float, model.feature_importances_)))}


# ---------------------------------------------------------------------------
# Task 2: cloud-cover multiple regression
# ---------------------------------------------------------------------------
def train_cloud_regressors(df):
    X = df[config.CLOUD_FEATURES]          # Cloud_Cover is NOT in here
    y = df[config.CLOUD_TARGET]
    assert config.CLOUD_TARGET not in X.columns

    # One split, shared by both regressors => a fair comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    scoring = {"neg_rmse": "neg_root_mean_squared_error", "neg_mae": "neg_mean_absolute_error", "r2": "r2"}

    candidates = {
        "linear_regression": {
            "name": "Multiple Linear Regression",
            "description": (
                "Fits Cloud_Cover = b0 + b1·Temperature + b2·Humidity + b3·Wind_Speed + b4·Pressure "
                "by ordinary least squares. It is the linear baseline: easy to interpret, but it "
                "can only capture straight-line (additive) relationships."
            ),
            "estimator": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            "grid": {},  # nothing to tune
        },
        "random_forest": {
            "name": "Random Forest Regressor",
            "description": (
                "Averages many regression trees, so it can model non-linear relationships and "
                "interactions between features without them being specified by hand. More "
                "flexible, but also more prone to fitting noise."
            ),
            "estimator": RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_STATE),
            "grid": {"max_depth": [None, 6], "min_samples_leaf": [5, 20]},
        },
    }

    results, fitted = {}, {}
    for key, spec in candidates.items():
        print(f"  [cloud] tuning {spec['name']} with {config.CV_FOLDS}-fold CV ...")
        search = GridSearchCV(spec["estimator"], spec["grid"], scoring=scoring, refit="neg_rmse",
                              cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        fitted[key] = model

        test_pred = model.predict(X_test)
        results[key] = {
            "name": spec["name"],
            "description": spec["description"],
            "best_params": dict(search.best_params_),
            "cv": _cv_summary(search, ["neg_rmse", "neg_mae", "r2"]),
            "train": regression_metrics(y_train, model.predict(X_train)),
            "test": regression_metrics(y_test, test_pred),
            "actual_vs_predicted": {
                "actual": [round(float(v), 3) for v in y_test],
                "predicted": [round(float(v), 3) for v in test_pred],
            },
            "interpretation": _cloud_model_interpretation(key, model),
        }

    selected, reason = _one_se_choice(results, "rmse", False, simple_key="linear_regression")

    # Reference point: always predict the training mean (R² ≈ 0 by construction).
    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline = regression_metrics(y_test, dummy.predict(X_test))
    baseline["constant_prediction"] = float(dummy.constant_[0][0])

    summary = {
        "features": config.CLOUD_FEATURES,
        "target": config.CLOUD_TARGET,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "cv_folds": config.CV_FOLDS,
        "selection_metric": "Mean 5-fold cross-validated RMSE (training data only)",
        "selection_rule": (
            "Lowest mean CV RMSE; if the simpler Multiple Linear Regression is within one "
            "standard error of the best, prefer it. The test set is NOT used for this choice."
        ),
        "selected": selected,
        "selection_reason": reason,
        "output_policy": (
            "Evaluation metrics use the raw, unclipped model outputs. On the prediction page the "
            "value shown is limited to the 0–100 scale of the target column, and the raw model "
            "output is always shown next to it whenever limiting was applied."
        ),
        "mean_baseline": baseline,
        # Pearson correlation of each input with the target (training data only).
        "train_correlations": {c: float(X_train[c].corr(y_train)) for c in config.CLOUD_FEATURES},
        "models": results,
    }
    bundle = {
        "format_version": config.ARTIFACT_FORMAT_VERSION,
        "task": "cloud_regression",
        "sklearn_version": sklearn.__version__,
        "trained_at": _now(),
        "features": config.CLOUD_FEATURES,
        "target": config.CLOUD_TARGET,
        "training_ranges": _training_ranges(X_train),
        "selected": selected,
        "models": {k: {"name": results[k]["name"], "estimator": fitted[k]} for k in fitted},
    }
    return bundle, summary


def _cloud_model_interpretation(key, model):
    if key == "linear_regression":
        scaler, lr = model.named_steps["scaler"], model.named_steps["model"]
        # Convert standardised coefficients back to the original feature units so
        # the equation can be written directly in terms of the raw inputs.
        raw_coef = lr.coef_ / scaler.scale_
        intercept = float(lr.intercept_ - np.sum(raw_coef * scaler.mean_))
        return {"type": "linear_equation",
                "intercept": intercept,
                "coefficients": dict(zip(config.CLOUD_FEATURES, map(float, raw_coef))),
                "standardised_coefficients": dict(zip(config.CLOUD_FEATURES, map(float, lr.coef_)))}
    return {"type": "feature_importances",
            "values": dict(zip(config.CLOUD_FEATURES, map(float, model.feature_importances_)))}


# ---------------------------------------------------------------------------
def main() -> int:
    print(f"Loading dataset from {config.DATA_PATH.relative_to(config.PROJECT_ROOT)} ...")
    try:
        df = load_dataset()
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"  {len(df)} valid rows.")

    rain_bundle, rain_summary = train_rain_classifiers(df)
    cloud_bundle, cloud_summary = train_cloud_regressors(df)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sha = file_sha256()
    rain_bundle["dataset_sha256"] = cloud_bundle["dataset_sha256"] = sha
    joblib.dump(rain_bundle, config.RAIN_MODEL_PATH)
    joblib.dump(cloud_bundle, config.CLOUD_MODEL_PATH)

    metrics = {
        "generated_at": _now(),
        "sklearn_version": sklearn.__version__,
        "random_state": config.RANDOM_STATE,
        "test_size": config.TEST_SIZE,
        "dataset_sha256": sha,
        "dataset": summarize_dataset(df),
        "classification": rain_summary,
        "regression": cloud_summary,
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    rs, cs = rain_summary, cloud_summary
    print("\nRain classification (held-out test set):")
    for key, m in rs["models"].items():
        t = m["test"]
        print(f"  {m['name']:<28} acc={t['accuracy']:.3f}  rain P={t['precision_rain']:.3f} "
              f"R={t['recall_rain']:.3f} F1={t['f1_rain']:.3f}  AUC={t['roc_auc']:.3f}")
    print(f"  -> selected: {rs['models'][rs['selected']]['name']} ({rs['selection_reason']})")
    print("\nCloud-cover regression (held-out test set):")
    for key, m in cs["models"].items():
        t = m["test"]
        print(f"  {m['name']:<28} R2={t['r2']:.4f}  MAE={t['mae']:.3f}  RMSE={t['rmse']:.3f}")
    b = cs["mean_baseline"]
    print(f"  {'(mean baseline)':<28} R2={b['r2']:.4f}  MAE={b['mae']:.3f}  RMSE={b['rmse']:.3f}")
    print(f"  -> selected: {cs['models'][cs['selected']]['name']} ({cs['selection_reason']})")
    print(f"\nSaved models and metrics to {config.MODELS_DIR.relative_to(config.PROJECT_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
