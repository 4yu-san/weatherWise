"""Training pipeline, evaluation consistency and artifact persistence."""
import joblib
import numpy as np
import pandas as pd
import pytest

from ml import config
from ml.evaluate import rain_decision, rain_probability
from ml.predict import ModelArtifactError, load_bundle


def test_regression_never_uses_cloud_cover_as_input(trained):
    bundle, summary = trained["cloud"]
    assert config.CLOUD_TARGET not in bundle["features"]
    assert bundle["features"] == ["Temperature", "Humidity", "Wind_Speed", "Pressure"]
    for entry in bundle["models"].values():
        assert list(entry["estimator"].feature_names_in_) == bundle["features"]


def test_two_classifiers_and_two_regressors(trained):
    assert set(trained["rain"][0]["models"]) == {"logistic_regression", "random_forest"}
    assert set(trained["cloud"][0]["models"]) == {"linear_regression", "random_forest"}


def test_regressors_share_the_same_test_split(trained):
    models = trained["cloud"][1]["models"]
    a = models["linear_regression"]["actual_vs_predicted"]["actual"]
    b = models["random_forest"]["actual_vs_predicted"]["actual"]
    assert a == b and len(a) == trained["cloud"][1]["n_test"]


def test_classification_split_is_stratified(trained):
    s = trained["rain"][1]
    assert s["train_rain_share"] == pytest.approx(s["test_rain_share"], abs=0.01)


def test_rain_probability_uses_rain_column(trained, df):
    bundle, _ = trained["rain"]
    X = df[config.RAIN_FEATURES].head(50)
    for entry in bundle["models"].values():
        model = entry["estimator"]
        rain_idx = list(model.classes_).index("rain")
        np.testing.assert_allclose(rain_probability(model, X), model.predict_proba(X)[:, rain_idx])


def test_threshold_labels_match_model_predict_at_0_5(trained, df):
    """At the 0.5 threshold our labels must equal scikit-learn's own predict()."""
    assert config.RAIN_DECISION_THRESHOLD == 0.5
    bundle, _ = trained["rain"]
    X = df[config.RAIN_FEATURES]
    for entry in bundle["models"].values():
        model = entry["estimator"]
        ours = rain_decision(rain_probability(model, X), 0.5)
        assert (ours == model.predict(X)).mean() > 0.999  # exact ties at 0.5 are the only possible difference


def test_metrics_are_consistent(trained):
    s = trained["rain"][1]
    for m in s["models"].values():
        t = m["test"]
        assert np.sum(t["confusion_matrix"]["matrix"]) == s["n_test"]
        (tn, fp), (fn, tp) = t["confusion_matrix"]["matrix"]
        assert t["recall_rain"] == pytest.approx(tp / (tp + fn))
        assert t["accuracy"] == pytest.approx((tp + tn) / s["n_test"])
        for k in ("accuracy", "precision_rain", "recall_rain", "f1_rain", "roc_auc"):
            assert 0.0 <= t[k] <= 1.0
    r = trained["cloud"][1]
    for m in r["models"].values():
        t = m["test"]
        assert t["rmse"] >= t["mae"] >= 0
        assert t["n_samples"] == r["n_test"]


def test_model_selection_uses_known_keys(trained):
    assert trained["rain"][1]["selected"] in trained["rain"][0]["models"]
    assert trained["cloud"][1]["selected"] in trained["cloud"][0]["models"]


def test_artifact_save_and_load_roundtrip(trained, df, tmp_path):
    bundle, _ = trained["cloud"]
    path = tmp_path / "cloud.joblib"
    joblib.dump(bundle, path)
    loaded = load_bundle(path, "cloud_regression")
    X = df[config.CLOUD_FEATURES].head(20)
    for key, entry in bundle["models"].items():
        np.testing.assert_allclose(loaded["models"][key]["estimator"].predict(X), entry["estimator"].predict(X))


def test_missing_artifact_detected(tmp_path):
    with pytest.raises(ModelArtifactError, match="train"):
        load_bundle(tmp_path / "missing.joblib", "rain_classification")


def test_incompatible_artifact_detected(trained, tmp_path):
    bundle = dict(trained["rain"][0], sklearn_version="0.0.1")
    path = tmp_path / "old.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ModelArtifactError, match="scikit-learn"):
        load_bundle(path, "rain_classification")


def test_wrong_task_artifact_detected(trained, tmp_path):
    path = tmp_path / "rain.joblib"
    joblib.dump(trained["rain"][0], path)
    with pytest.raises(ModelArtifactError):
        load_bundle(path, "cloud_regression")


def test_training_is_reproducible(trained, df):
    from ml.train_models import train_cloud_regressors
    _, again = train_cloud_regressors(df)
    first = trained["cloud"][1]["models"]["linear_regression"]["test"]["rmse"]
    assert again["models"]["linear_regression"]["test"]["rmse"] == pytest.approx(first)


def test_corrupt_artifact_detected(tmp_path):
    path = tmp_path / "bad.joblib"
    path.write_bytes(b"not a joblib file")
    with pytest.raises(ModelArtifactError):
        load_bundle(path, "rain_classification")
