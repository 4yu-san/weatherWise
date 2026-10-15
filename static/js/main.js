/* Shared helpers used on every page. */
(function () {
  "use strict";

  // Mobile navigation toggle.
  const toggle = document.querySelector(".nav-toggle");
  const nav = document.getElementById("site-nav");
  if (toggle && nav) {
    toggle.addEventListener("click", () => {
      const open = nav.classList.toggle("open");
      toggle.setAttribute("aria-expanded", String(open));
    });
  }

  /**
   * Wire up a prediction form (used by the rain and cloud-cover pages).
   *   form     – the <form> element (data-endpoint holds the API URL)
   *   panel    – the result <section>
   *   render   – function(result, panel) that fills in the result body
   * Prediction happens without reloading the page; the form can be
   * re-submitted as many times as the user likes.
   */
  function setupPredictionForm(form, panel, render) {
    if (!form || !panel) return;
    const endpoint = form.dataset.endpoint;
    const submitBtn = form.querySelector('button[type="submit"]');
    const formError = form.querySelector(".form-error");
    const parts = {
      empty: panel.querySelector(".result-empty"),
      loading: panel.querySelector(".result-loading"),
      body: panel.querySelector(".result-body"),
    };

    function show(which) {
      for (const [name, el] of Object.entries(parts)) el.hidden = name !== which;
      panel.setAttribute("aria-busy", String(which === "loading"));
    }

    function clearErrors() {
      formError.hidden = true;
      formError.textContent = "";
      form.querySelectorAll("input").forEach((input) => {
        input.removeAttribute("aria-invalid");
        const err = document.getElementById("e-" + input.name);
        if (err) err.textContent = "";
      });
    }

    function showFieldErrors(fields) {
      let first = null;
      for (const [name, msg] of Object.entries(fields || {})) {
        const input = form.querySelector(`[name="${CSS.escape(name)}"]`);
        const err = document.getElementById("e-" + name);
        if (input && err) {
          input.setAttribute("aria-invalid", "true");
          err.textContent = `${name.replace("_", " ")} ${msg}.`;
          first = first || input;
        } else {
          formError.hidden = false;
          formError.textContent += (formError.textContent ? " " : "") + msg;
        }
      }
      if (first) first.focus();
    }

    // Quick client-side check for empty / non-numeric values; the server
    // performs the authoritative validation (ranges, finiteness, etc.).
    function collect() {
      const payload = {};
      const errors = {};
      form.querySelectorAll("input").forEach((input) => {
        const raw = input.value.trim();
        if (raw === "") errors[input.name] = "is required";
        else if (!Number.isFinite(Number(raw))) errors[input.name] = "must be a number";
        else payload[input.name] = Number(raw);
      });
      const model = form.querySelector("select[name=model]");
      if (model) payload.model = model.value;
      return { payload, errors };
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      clearErrors();
      const { payload, errors } = collect();
      if (Object.keys(errors).length) {
        showFieldErrors(errors);
        return;
      }

      show("loading");
      submitBtn.disabled = true;
      submitBtn.classList.add("is-loading");
      try {
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        let data = null;
        try { data = await res.json(); } catch (_) { /* non-JSON response */ }
        if (!res.ok || !data || !data.ok) {
          show("empty");
          formError.hidden = false;
          formError.textContent = (data && data.error) || `Request failed (HTTP ${res.status}).`;
          if (data && data.fields) showFieldErrors(data.fields);
          return;
        }
        render(data, panel);
        show("body");
      } catch (err) {
        show("empty");
        formError.hidden = false;
        formError.textContent = "Could not reach the server. Is the Flask app still running?";
      } finally {
        submitBtn.disabled = false;
        submitBtn.classList.remove("is-loading");
      }
    });

    form.addEventListener("reset", () => {
      clearErrors();
      show("empty");
    });
  }

  function fillWarnings(list, warnings) {
    list.replaceChildren(...(warnings || []).map((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      return li;
    }));
  }

  window.WeatherWise = { setupPredictionForm, fillWarnings };
})();
