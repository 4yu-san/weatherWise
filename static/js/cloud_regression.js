/* Cloud-cover regression page: send inputs to /api/predict-cloud-cover. */
(function () {
  "use strict";
  const form = document.getElementById("cloud-form");
  const panel = document.getElementById("cloud-result");

  function render(result, panel) {
    const out = (key) => panel.querySelector(`[data-out="${key}"]`);
    const value = result.prediction;

    out("value").textContent = value.toFixed(1);
    out("meter").style.width = `${value}%`;
    out("meter-label").setAttribute("aria-label", `Predicted cloud cover ${value.toFixed(1)} on a 0 to 100 scale`);
    out("model").textContent = result.model_name;
    out("raw").textContent = result.raw_prediction.toFixed(3) + (result.was_limited ? " (outside 0–100, limited for display)" : "");

    let explanation = `${result.model_name} estimates Cloud_Cover from Temperature, Humidity, ` +
      `Wind_Speed and Pressure only.`;
    if (result.test_r2 !== null && result.test_r2 !== undefined) {
      out("quality").textContent = `R² ${result.test_r2.toFixed(4)} · RMSE ${result.test_rmse.toFixed(2)}`;
      explanation += ` On unseen test data its typical error (RMSE) was about ${result.test_rmse.toFixed(1)} points.`;
      if (result.test_r2 < 0.05) {
        explanation += " Because its test R² is close to zero, the inputs explain almost none of the variation " +
          "in cloud cover, so this estimate is close to the dataset average and should not be relied on.";
      }
    } else {
      out("quality").textContent = "not available";
    }
    out("explanation").textContent = explanation;
    WeatherWise.fillWarnings(out("warnings"), result.warnings);
  }

  WeatherWise.setupPredictionForm(form, panel, render);
})();
