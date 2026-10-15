# WeatherWise: Weather Prediction & ML Analysis

An academic machine-learning mini-project with a local Flask website. It covers two supervised learning tasks on one supplied dataset:

1. **Rain prediction (classification).** Predicts `rain` / `no rain` and the estimated probability of rain from Temperature, Humidity, Wind_Speed, Cloud_Cover and Pressure. Compares **Logistic Regression** and a **Random Forest Classifier**.
2. **Cloud-cover multiple regression.** `Y = Cloud_Cover`, `X = Temperature, Humidity, Wind_Speed, Pressure`. Compares **Multiple Linear Regression** and a **Random Forest Regressor**.

Every prediction comes from a saved, trained model. Every metric and chart comes from real predictions on a held-out test set. No figures are hard-coded.

> **Disclaimer:** this is an educational project, not a weather forecasting service. It predicts from values you type in and does not fetch live weather.

---

## Requirements

- **Python 3.9 or newer.** Tested on Linux with Python 3.9, 3.10, 3.12 and 3.14 (fresh install from `requirements.txt` plus the full test suite). Not yet tested on Windows itself.
- Internet access for the one-time `pip install`. No API keys, databases or internet connection are needed once dependencies are installed. After that nothing needs internet. Chart.js is bundled in `static/vendor/`.

## Setup and run

### Windows: quick start (double-click)

Double-click **`start.bat`** in the project folder. The first run creates a `.venv`, installs the packages, **trains the models on that computer** and opens the site in your browser. Every later run just starts the server. Close the window to stop it.

If it fails, the window stays open with a message. Run the manual PowerShell steps below and see Troubleshooting.

### Linux / macOS

```bash
cd weatherWise
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.train_models      # trains models, writes models/*.joblib + metrics JSON
python app.py                  # starts the server
```

### Windows (PowerShell, manual steps)

```powershell
cd weatherWise
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m ml.train_models
python app.py
```

(Using `cmd.exe`? Activate with `.venv\Scripts\activate.bat` instead.)

Then open **http://127.0.0.1:5000** in your browser. Stop the server with `Ctrl+C`.

### Run the tests

```bash
python -m pytest -q
```

## Dataset

- Location: **`data/Weather_classification_Logistic.csv`** (keep this exact file name).
- 2,500 rows and 6 columns: `Temperature, Humidity, Wind_Speed, Cloud_Cover, Pressure, Rain`.
- `Rain` has the labels `rain` (314, 12.56%) and `no rain` (2,186, 87.44%), so the classes are imbalanced. No values are missing.
- **Provenance and units are not documented.** The site therefore shows no units. The data should not be treated as real weather observations until its source is verified.
- The loader (`ml/data.py`) checks columns, numeric and finite values and allowed labels. If the file is missing or invalid, you get a clear error. Replacement data is never generated.

### Retraining after changing the CSV

Replace the CSV (same columns and labels), then run `python -m ml.train_models` again. The running server picks up the new models and metrics automatically, with no restart needed. Retrain as well after upgrading scikit-learn, because saved models are tied to the version that created them, and the app tells you when they don't match.

## Project structure

```text
weatherWise/
├── app.py                    Flask routes (pages + JSON API); never trains
├── start.bat                 Windows one-click setup + start
├── requirements.txt
├── data/Weather_classification_Logistic.csv
├── models/                   created by the training script
│   ├── rain_classifier.joblib    both classifiers + selected key + metadata
│   ├── cloud_regressor.joblib    both regressors + selected key + metadata
│   └── evaluation_metrics.json   every number/chart shown on the site
├── ml/
│   ├── config.py             paths, column names, seed, threshold, input limits
│   ├── data.py               load / validate / summarise the CSV
│   ├── train_models.py       training + model selection + saving (run this)
│   ├── evaluate.py           metric functions (shared by training & inference)
│   └── predict.py            load artifacts, validate input, predict
├── templates/                Jinja2 pages (base, index, rain, cloud, evaluation)
├── static/                   CSS, JS, bundled Chart.js, icon
└── tests/                    pytest: data, models, API
```

## ML methodology (summary for the viva)

| | Rain classification | Cloud-cover regression |
|---|---|---|
| Target | `Rain` (positive class `rain`) | `Cloud_Cover` |
| Features | all 5 columns | Temperature, Humidity, Wind_Speed, Pressure |
| Split | 80:20, **stratified**, seed 42 | 80:20, seed 42, same split for both models |
| Model 1 | Logistic Regression (StandardScaler in a Pipeline, `class_weight="balanced"`) | Multiple Linear Regression (StandardScaler + LinearRegression) |
| Model 2 | Random Forest Classifier (`class_weight="balanced"`) | Random Forest Regressor |
| Tuning | GridSearchCV, 5-fold stratified CV on training data | GridSearchCV, 5-fold CV on training data |
| Selection metric | mean CV F1 of the `rain` class | mean CV RMSE |
| Test metrics | accuracy, balanced accuracy, precision/recall/F1 (rain + macro), ROC-AUC, Brier score, confusion matrix, classification report | R², MAE, RMSE (on raw, unclipped predictions) |

Key decisions:

- **No leakage.** Scalers sit inside scikit-learn Pipelines, so they are fitted on training folds only. The test set plays no part in tuning or model choice. Each model is scored on it once, at the end.
- **Model selection uses the one-standard-error rule.** The model with the best mean CV score wins, unless the simpler model (Logistic / Linear Regression) is within one standard error of it. In that case the simpler model is preferred.
- **Class imbalance** is handled with `class_weight="balanced"`. The test set is never resampled.
- **Rain probability** uses `predict_proba`. The `rain` column is found through `model.classes_`, never assumed. The label is `rain` when the probability is at or above the **0.5** threshold. That is the same rule used for the reported test metrics, and it matches scikit-learn's `predict()` (a test checks this). No other threshold was tuned.
- **Cloud-cover output policy.** Metrics always use raw predictions. On the prediction page, a value outside 0–100 would be limited to that range for display, and the raw value is shown beside it.
- **Out-of-range inputs** (valid but outside the training data range) are accepted with an extrapolation warning, never silently clipped. Impossible values (for example humidity above 100 or negative wind speed) are rejected.

### Findings to expect

The training script prints its results. With the supplied CSV:

- The **Random Forest Classifier** is selected. It is near-perfect on the test set, while Logistic Regression has high rain recall but much lower precision. The evaluation page's "dataset audit" shows why: a depth-3 decision tree recovers the label almost exactly using simple thresholds (Humidity > ~70, Cloud_Cover > ~50, Temperature < ~25). This strongly suggests the dataset is **synthetic**, so these scores will not carry over to real weather.
- **Cloud-cover regression has R² ≈ 0** for both models, and neither beats predicting the mean. The four inputs are essentially uncorrelated with `Cloud_Cover` in this data. **Multiple Linear Regression** is selected (lowest CV RMSE). The Random Forest's higher training R² than test R² is an example of overfitting. These results are reported as they are, without any tuning on the test set.

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/rain-prediction` | Rain prediction page |
| GET | `/cloud-regression` | Cloud-cover regression page |
| GET | `/model-evaluation` | Evaluation dashboard |
| POST | `/api/predict-rain` | JSON `{Temperature, Humidity, Wind_Speed, Cloud_Cover, Pressure, model?}` |
| POST | `/api/predict-cloud-cover` | JSON `{Temperature, Humidity, Wind_Speed, Pressure, model?}` |
| GET | `/api/model-metrics` | saved evaluation metrics |
| GET | `/api/dataset-summary` | dataset statistics |

`model` is optional (`logistic_regression` / `random_forest`, or `linear_regression` / `random_forest`). It defaults to the selected model. Invalid input returns HTTP 400 with per-field messages. Missing or incompatible models return 503 with retraining instructions. Stack traces are never sent to the browser.

Example:

```bash
curl -X POST http://127.0.0.1:5000/api/predict-rain \
  -H "Content-Type: application/json" \
  -d '{"Temperature": 18, "Humidity": 85, "Wind_Speed": 5, "Cloud_Cover": 75, "Pressure": 1010}'
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "Trained model not found" / empty dashboards | Run `python -m ml.train_models` from the `weatherWise` folder. |
| "trained with scikit-learn X but Y is installed" | Normal on a new computer. Retrain with `python -m ml.train_models` (takes seconds). `start.bat` and the tests do this automatically. |
| "Dataset not found" | Put the CSV at `data/Weather_classification_Logistic.csv`. |
| `ModuleNotFoundError: ml` | Run commands from inside the `weatherWise` folder, using `python -m ml.train_models` (not `python ml/train_models.py`). |
| `ModuleNotFoundError: flask` / `sklearn` | Activate the virtual environment first, then `pip install -r requirements.txt`. |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use `cmd.exe` with `activate.bat`. |
| `start.bat` says Python was not found, or opens the Microsoft Store | Install Python from python.org (the “Install just for me” option needs no admin rights) and tick “Add python.exe to PATH”. In Windows Settings → Apps → Advanced app settings → App execution aliases, switch off the “python.exe” Store alias. |
| `pip install` fails or times out (proxy / blocked network) | Ask IT for the proxy, then run `.venv\Scripts\python.exe -m pip install --proxy http://user:pass@host:port -r requirements.txt`, or install the packages at home and copy the whole folder (including `.venv`) only if both PCs have the same Python version and path. |
| `start.bat` is blocked or closes instantly | Open `cmd.exe`, `cd` to the folder and run `start.bat` to see the error message. If .bat files are blocked, use the manual PowerShell steps. |
| Browser shows “can't connect” right after start | Wait a few seconds and refresh, or open http://127.0.0.1:5000 yourself. |
| Port 5000 already in use (e.g. macOS AirPlay) | Change `port=5000` at the bottom of `app.py`, e.g. to `5050`. |

## Security note

Model files are loaded with `joblib`, which can execute code when loading. The app only loads its own fixed files from `models/`, which are created by the training script. Never replace them with files from an untrusted source.
