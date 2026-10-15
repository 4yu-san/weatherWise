"""Flask pages and JSON API."""
import pytest

RAIN_INPUT = {"Temperature": 18, "Humidity": 85, "Wind_Speed": 5, "Cloud_Cover": 75, "Pressure": 1010}
CLOUD_INPUT = {"Temperature": 22, "Humidity": 60, "Wind_Speed": 10, "Pressure": 1015}


@pytest.mark.parametrize("url", ["/", "/rain-prediction", "/cloud-regression", "/model-evaluation"])
def test_pages_load(client, url):
    res = client.get(url)
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    for link in ["/rain-prediction", "/cloud-regression", "/model-evaluation"]:
        assert f'href="{link}"' in html  # navigation present on every page


def test_unknown_page_is_404(client):
    assert client.get("/does-not-exist").status_code == 404
    res = client.get("/api/does-not-exist")
    assert res.status_code == 404 and res.get_json()["ok"] is False


def test_predict_rain(client):
    res = client.post("/api/predict-rain", json=RAIN_INPUT)
    assert res.status_code == 200
    data = res.get_json()
    assert 0.0 <= data["probability_rain"] <= 1.0
    assert data["prediction"] in ("rain", "no rain")
    # Label must agree with the probability at the configured threshold.
    assert (data["prediction"] == "rain") == (data["probability_rain"] >= data["threshold"])
    assert data["model_name"]


def test_predict_rain_each_model(client):
    for key in ("logistic_regression", "random_forest"):
        data = client.post("/api/predict-rain", json={**RAIN_INPUT, "model": key}).get_json()
        assert data["ok"] and data["model_key"] == key


def test_repeated_predictions(client):
    results = [client.post("/api/predict-rain", json=RAIN_INPUT).get_json() for _ in range(5)]
    assert len({r["probability_rain"] for r in results}) == 1


def test_predict_cloud_cover(client):
    res = client.post("/api/predict-cloud-cover", json=CLOUD_INPUT)
    assert res.status_code == 200
    data = res.get_json()
    assert 0.0 <= data["prediction"] <= 100.0
    assert set(data["inputs"]) == set(CLOUD_INPUT)
    assert data["test_r2"] is not None


def test_cloud_cover_input_rejected_for_regression(client):
    res = client.post("/api/predict-cloud-cover", json={**CLOUD_INPUT, "Cloud_Cover": 50})
    assert res.status_code == 400
    assert "Cloud_Cover" in res.get_json()["fields"]


@pytest.mark.parametrize("bad, field", [
    ({"Humidity": None}, "Humidity"),
    ({"Humidity": "abc"}, "Humidity"),
    ({"Humidity": 150}, "Humidity"),
    ({"Wind_Speed": -3}, "Wind_Speed"),
    ({"Pressure": True}, "Pressure"),
    ({"Temperature": float("inf")}, "Temperature"),
])
def test_invalid_rain_inputs(client, bad, field):
    res = client.post("/api/predict-rain", json={**RAIN_INPUT, **bad})
    assert res.status_code == 400
    body = res.get_json()
    assert body["ok"] is False and field in body["fields"]


def test_missing_field(client):
    payload = dict(RAIN_INPUT)
    del payload["Pressure"]
    res = client.post("/api/predict-rain", json=payload)
    assert res.status_code == 400 and "Pressure" in res.get_json()["fields"]


def test_non_json_body(client):
    res = client.post("/api/predict-rain", data="hello", content_type="text/plain")
    assert res.status_code == 400


def test_unknown_model_key(client):
    res = client.post("/api/predict-rain", json={**RAIN_INPUT, "model": "svm"})
    assert res.status_code == 400 and "model" in res.get_json()["fields"]


def test_out_of_training_range_warns_but_does_not_clip(client):
    data = client.post("/api/predict-rain", json={**RAIN_INPUT, "Temperature": 45}).get_json()
    assert data["ok"] and data["inputs"]["Temperature"] == 45
    assert any("outside the training range" in w for w in data["warnings"])


def test_get_on_predict_endpoint_not_allowed(client):
    assert client.get("/api/predict-rain").status_code == 405


def test_metrics_and_summary_endpoints(client):
    m = client.get("/api/model-metrics").get_json()
    assert m["ok"] and "classification" in m["metrics"] and "regression" in m["metrics"]
    s = client.get("/api/dataset-summary").get_json()
    assert s["ok"] and s["summary"]["rows"] == 2500
