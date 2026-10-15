"""Shared pytest fixtures."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml import config  # noqa: E402
from ml.data import load_dataset  # noqa: E402


@pytest.fixture(scope="session")
def df():
    return load_dataset()


@pytest.fixture(scope="session")
def trained():
    """Train both tasks once per test session (in memory, nothing written)."""
    from ml.train_models import train_cloud_regressors, train_rain_classifiers
    data = load_dataset()
    rain_bundle, rain_summary = train_rain_classifiers(data)
    cloud_bundle, cloud_summary = train_cloud_regressors(data)
    return {"rain": (rain_bundle, rain_summary), "cloud": (cloud_bundle, cloud_summary)}


@pytest.fixture(scope="session")
def ensure_artifacts():
    """The API tests need loadable saved models. If they are missing, or were saved
    by a different scikit-learn version than the one installed (the case on a new
    machine), retrain them first."""
    from ml.predict import ModelArtifactError, load_bundle
    try:
        load_bundle(config.RAIN_MODEL_PATH, "rain_classification")
        load_bundle(config.CLOUD_MODEL_PATH, "cloud_regression")
        assert config.METRICS_PATH.exists()
    except (ModelArtifactError, AssertionError):
        from ml.train_models import main
        assert main() == 0


@pytest.fixture()
def client(ensure_artifacts):
    from app import app
    app.config.update(TESTING=True)
    return app.test_client()
