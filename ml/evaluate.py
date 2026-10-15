"""Metric helpers used by the training script.

Every number shown on the website is produced by one of these functions from
real model predictions on real held-out data.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
)

from ml import config


def rain_probability(model, X) -> np.ndarray:
    """Estimated probability of the "rain" class.

    The column is looked up from model.classes_ rather than assuming column 1,
    so the code stays correct even if the label order ever changes.
    """
    classes = list(model.classes_)
    rain_col = classes.index(config.POSITIVE_LABEL)
    return model.predict_proba(X)[:, rain_col]


def rain_decision(probabilities, threshold: float = config.RAIN_DECISION_THRESHOLD) -> np.ndarray:
    """Turn rain probabilities into class labels at the configured threshold."""
    negative = [lbl for lbl in config.RAIN_LABELS if lbl != config.POSITIVE_LABEL][0]
    return np.where(np.asarray(probabilities) >= threshold, config.POSITIVE_LABEL, negative)


def classification_metrics(model, X, y_true, threshold: float = config.RAIN_DECISION_THRESHOLD) -> dict:
    """Full classification evaluation of a fitted model on (X, y_true)."""
    proba = rain_probability(model, X)
    y_pred = rain_decision(proba, threshold)
    pos = config.POSITIVE_LABEL
    labels = config.RAIN_LABELS  # fixed order: ["no rain", "rain"]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    y_binary = (np.asarray(y_true) == pos).astype(int)
    roc_auc = None
    roc = None
    if len(np.unique(y_binary)) == 2:  # ROC-AUC is only defined if both classes are present
        roc_auc = float(roc_auc_score(y_binary, proba))
        fpr, tpr, _ = roc_curve(y_binary, proba)
        roc = {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]}

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # Minority ("rain") class metrics – the most informative ones here.
        "precision_rain": float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        "recall_rain": float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        "f1_rain": float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0)),
        # Macro average = unweighted mean over both classes.
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "roc_auc": roc_auc,
        "brier_score": float(brier_score_loss(y_binary, proba)),
        "confusion_matrix": {
            "labels": labels,  # rows = actual, columns = predicted
            "matrix": cm.tolist(),
        },
        "classification_report": report,
        "roc_curve": roc,
        "n_samples": int(len(y_true)),
    }


def regression_metrics(y_true, y_pred) -> dict:
    """R², MAE and RMSE on the RAW model outputs (never clipped)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo, hi = config.CLOUD_COVER_RANGE
    outside = int(((y_pred < lo) | (y_pred > hi)).sum())
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "prediction_min": float(y_pred.min()),
        "prediction_max": float(y_pred.max()),
        "n_outside_0_100": outside,
        "n_samples": int(len(y_true)),
    }
