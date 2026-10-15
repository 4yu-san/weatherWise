/* Evaluation dashboard charts. All data comes from /api/model-metrics,
   i.e. the metrics JSON written by the training script. Nothing is invented. */
(function () {
  "use strict";
  if (typeof Chart === "undefined") return;

  const charts = [];
  let metrics = null;

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function theme() {
    return {
      a: css("--series-a"), b: css("--series-b"), base: css("--series-base"),
      text: css("--text-2"), grid: css("--grid"), surface: css("--surface-2"), ink: css("--text"),
    };
  }

  function baseOptions(t) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: { labels: { color: t.text, boxWidth: 12, boxHeight: 12, useBorderRadius: true, borderRadius: 3 } },
        tooltip: { backgroundColor: t.ink, titleColor: t.surface, bodyColor: t.surface, padding: 10 },
      },
      scales: {
        x: { ticks: { color: t.text }, grid: { color: t.grid }, border: { color: t.grid } },
        y: { ticks: { color: t.text }, grid: { color: t.grid }, border: { color: t.grid } },
      },
    };
  }

  function bar(color) {
    return { backgroundColor: color, borderRadius: 4, borderSkipped: "start",
             borderColor: "transparent", borderWidth: 0, maxBarThickness: 46 };
  }

  function make(id, config) {
    const el = document.getElementById(id);
    if (el) charts.push(new Chart(el, config));
  }

  function fmt(v, d) { return v === null || v === undefined ? "n/a" : Number(v).toFixed(d); }

  function draw() {
    charts.splice(0).forEach((c) => c.destroy());
    const t = theme();
    const c = metrics.classification;
    const r = metrics.regression;
    const lr = c.models.logistic_regression, rfc = c.models.random_forest;
    const lin = r.models.linear_regression, rfr = r.models.random_forest;

    // --- Classification: rain-class metric comparison ---------------------
    const clsLabels = ["Precision (rain)", "Recall (rain)", "F1 (rain)", "ROC-AUC"];
    const clsValues = (m) => [m.test.precision_rain, m.test.recall_rain, m.test.f1_rain, m.test.roc_auc];
    let opts = baseOptions(t);
    opts.scales.y.min = 0; opts.scales.y.max = 1;
    opts.plugins.tooltip.callbacks = { label: (ctx) => `${ctx.dataset.label}: ${fmt(ctx.raw, 3)}` };
    make("chart-cls-compare", {
      type: "bar",
      data: { labels: clsLabels, datasets: [
        { label: lr.name, data: clsValues(lr), ...bar(t.a) },
        { label: rfc.name, data: clsValues(rfc), ...bar(t.b) },
      ] },
      options: opts,
    });

    // --- Classification: ROC curves --------------------------------------
    const rocData = (m) => m.test.roc_curve ? m.test.roc_curve.fpr.map((x, i) => ({ x, y: m.test.roc_curve.tpr[i] })) : [];
    opts = baseOptions(t);
    opts.scales.x = { ...opts.scales.x, type: "linear", min: 0, max: 1, title: { display: true, text: "False positive rate", color: t.text } };
    opts.scales.y = { ...opts.scales.y, min: 0, max: 1, title: { display: true, text: "True positive rate (recall)", color: t.text } };
    opts.plugins.tooltip.callbacks = { label: (ctx) => `${ctx.dataset.label}: FPR ${fmt(ctx.raw.x, 3)}, TPR ${fmt(ctx.raw.y, 3)}` };
    make("chart-roc", {
      type: "line",
      data: { datasets: [
        { label: `${lr.name} (AUC ${fmt(lr.test.roc_auc, 3)})`, data: rocData(lr), borderColor: t.a, backgroundColor: t.a, borderWidth: 2, pointRadius: 0, stepped: false },
        { label: `${rfc.name} (AUC ${fmt(rfc.test.roc_auc, 3)})`, data: rocData(rfc), borderColor: t.b, backgroundColor: t.b, borderWidth: 2, pointRadius: 0 },
        { label: "Random guess", data: [{ x: 0, y: 0 }, { x: 1, y: 1 }], borderColor: t.base, borderDash: [5, 5], borderWidth: 1.5, pointRadius: 0 },
      ] },
      options: opts,
    });

    // --- Regression: actual vs predicted ---------------------------------
    function avp(id, m, color) {
      const pts = m.actual_vs_predicted.actual.map((a, i) => ({ x: a, y: m.actual_vs_predicted.predicted[i] }));
      const all = pts.flatMap((p) => [p.x, p.y]);
      const lo = Math.min(0, ...all), hi = Math.max(100, ...all);
      const o = baseOptions(t);
      o.scales.x = { ...o.scales.x, type: "linear", min: Math.floor(lo), max: Math.ceil(hi), title: { display: true, text: "Actual Cloud_Cover", color: t.text } };
      o.scales.y = { ...o.scales.y, min: Math.floor(lo), max: Math.ceil(hi), title: { display: true, text: "Predicted Cloud_Cover", color: t.text } };
      o.plugins.tooltip.callbacks = { label: (ctx) => ctx.datasetIndex === 0
        ? `actual ${fmt(ctx.raw.x, 1)}, predicted ${fmt(ctx.raw.y, 1)}` : "perfect prediction (y = x)" };
      make(id, {
        type: "scatter",
        data: { datasets: [
          { label: `Test records (n = ${pts.length})`, data: pts, backgroundColor: color + "99", borderColor: color, borderWidth: 0, pointRadius: 3, pointHoverRadius: 5 },
          { type: "line", label: "Perfect prediction (y = x)", data: [{ x: lo, y: lo }, { x: hi, y: hi }], borderColor: t.ink, borderDash: [6, 5], borderWidth: 1.5, pointRadius: 0 },
        ] },
        options: o,
      });
    }
    avp("chart-avp-linear", lin, t.a);
    avp("chart-avp-rf", rfr, t.b);

    // --- Regression: error metrics (MAE, RMSE share one unit => one axis) -
    const base = r.mean_baseline;
    opts = baseOptions(t);
    opts.scales.y.beginAtZero = true;
    opts.plugins.tooltip.callbacks = { label: (ctx) => `${ctx.dataset.label}: ${fmt(ctx.raw, 3)}` };
    make("chart-reg-errors", {
      type: "bar",
      data: { labels: ["MAE", "RMSE"], datasets: [
        { label: lin.name, data: [lin.test.mae, lin.test.rmse], ...bar(t.a) },
        { label: rfr.name, data: [rfr.test.mae, rfr.test.rmse], ...bar(t.b) },
        { label: "Mean baseline", data: [base.mae, base.rmse], ...bar(t.base) },
      ] },
      options: opts,
    });

    // --- Regression: R² (separate chart: different scale from errors) ----
    const r2 = [lin.test.r2, rfr.test.r2, base.r2];
    opts = baseOptions(t);
    opts.plugins.legend.display = false;
    opts.plugins.tooltip.callbacks = { label: (ctx) => `R² = ${fmt(ctx.raw, 4)}` };
    const span = Math.max(0.05, ...r2.map(Math.abs));
    opts.scales.y.min = -Math.ceil(span * 100) / 100;
    opts.scales.y.max = Math.max(Math.ceil(span * 100) / 100, 0.05);
    make("chart-reg-r2", {
      type: "bar",
      data: { labels: [lin.name, rfr.name, "Mean baseline"], datasets: [
        { label: "R²", data: r2, backgroundColor: [t.a, t.b, t.base], borderRadius: 4, borderSkipped: false, maxBarThickness: 56 },
      ] },
      options: opts,
    });
  }

  fetch("/api/model-metrics")
    .then((res) => res.json())
    .then((data) => {
      if (!data.ok) throw new Error(data.error);
      metrics = data.metrics;
      draw();
      // Redraw with the right colours if the OS switches light/dark mode.
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
    })
    .catch((err) => {
      document.querySelectorAll(".chart-box").forEach((box) => {
        box.innerHTML = `<p class="notice notice-error">Charts could not be loaded: ${String(err.message || err).replace(/[<>&]/g, "")}</p>`;
      });
    });
})();
