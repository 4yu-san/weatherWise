"""Dataset loading and validation."""
import pandas as pd
import pytest

from ml import config
from ml.data import DatasetError, load_dataset, summarize_dataset


def test_supplied_csv_loads(df):
    assert len(df) == 2500
    assert list(df.columns) == config.ALL_FEATURES + [config.RAIN_TARGET]
    assert set(df[config.RAIN_TARGET]) == {"rain", "no rain"}
    assert df.isna().sum().sum() == 0


def test_summary_matches_data(df):
    s = summarize_dataset(df)
    assert s["rows"] == 2500
    assert s["missing_values"] == 0
    dist = s["target_distribution"]
    assert dist["rain"]["count"] + dist["no rain"]["count"] == 2500
    assert dist["rain"]["percent"] == pytest.approx(12.56, abs=0.01)
    assert s["units_verified"] is False


def _write(tmp_path, frame):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


def _valid_frame(n=60):
    rows = [{"Temperature": 20, "Humidity": 50, "Wind_Speed": 5, "Cloud_Cover": 40,
             "Pressure": 1010, "Rain": "rain" if i % 5 == 0 else "no rain"} for i in range(n)]
    return pd.DataFrame(rows)


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.csv")


def test_missing_column_rejected(tmp_path):
    path = _write(tmp_path, _valid_frame().drop(columns=["Pressure"]))
    with pytest.raises(DatasetError, match="Pressure"):
        load_dataset(path)


def test_unknown_label_rejected(tmp_path):
    frame = _valid_frame()
    frame.loc[0, "Rain"] = "snow"
    with pytest.raises(DatasetError, match="snow"):
        load_dataset(_write(tmp_path, frame))


def test_non_numeric_feature_rejected(tmp_path):
    frame = _valid_frame().astype({"Humidity": object})
    frame.loc[3, "Humidity"] = "very humid"
    with pytest.raises(DatasetError, match="Humidity"):
        load_dataset(_write(tmp_path, frame))


def test_rows_with_missing_values_are_dropped_and_counted(tmp_path):
    frame = _valid_frame()
    frame.loc[1, "Temperature"] = None
    loaded = load_dataset(_write(tmp_path, frame))
    assert len(loaded) == len(frame) - 1
    assert summarize_dataset(loaded)["missing_values"] == 1
