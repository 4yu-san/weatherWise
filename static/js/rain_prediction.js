/* Rain prediction page: send inputs to /api/predict-rain and show the result. */
(function () {
  "use strict";
  const form = document.getElementById("rain-form");
  const panel = document.getElementById("rain-result");

  function render(result, panel) {
    const out = (key) => panel.querySelector(`[data-out="${key}"]`);
    const pct = result.probability_rain * 100;
    const thr = result.threshold * 100;

    out("probability").textContent = `${pct.toFixed(1)}%`;
    out("meter").style.width = `${pct}%`;
    out("threshold-marker").style.left = `${thr}%`;
    out("threshold-label").textContent = `threshold ${thr.toFixed(0)}%`;
    out("meter-label").setAttribute("aria-label",
      `Estimated rain probability ${pct.toFixed(1)} percent; decision threshold ${thr.toFixed(0)} percent`);

    // The label comes straight from the server's prediction, which applies
    // the configured threshold to the model's rain probability.
    const label = out("label");
    label.textContent = result.display_label;
    label.className = "badge " + (result.prediction === "rain" ? "badge-rain" : "badge-norain");

    out("model").textContent = result.model_name;
    out("threshold").textContent = `${thr.toFixed(0)}% (probability of rain)`;
    out("interpretation").textContent = result.interpretation;
    WeatherWise.fillWarnings(out("warnings"), result.warnings);
  }

  WeatherWise.setupPredictionForm(form, panel, render);
})();
