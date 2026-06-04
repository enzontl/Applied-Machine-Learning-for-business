/* ─────────────────────────────────────────────────────────────────────────
   URBAN OPTIMIZER — frontend logic
   Pas de framework. Fetch API + DOM manipulation. SessionStorage pour partager
   le job entre landing → dashboard.
   ───────────────────────────────────────────────────────────────────────── */

const API = {
  cities: "/api/cities",
  profiles: "/api/profiles",
  createJob: "/api/jobs",
  jobStatus: (id) => `/api/jobs/${id}`,
};

// État partagé entre pages via sessionStorage
const SS_KEY_JOB = "uo:lastJob";

function fmtNumber(n, opts = {}) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return new Intl.NumberFormat("fr-FR", opts).format(n);
}

function fmtEur(n) {
  if (n === null || n === undefined || !isFinite(n)) return "—";
  const abs = Math.abs(n);
  const sign = n < 0 ? "−" : "";
  if (abs >= 1e9) return `${sign}${(abs / 1e9).toFixed(2)} G€`;
  if (abs >= 1e6) return `${sign}${(abs / 1e6).toFixed(1)} M€`;
  if (abs >= 1e3) return `${sign}${(abs / 1e3).toFixed(0)} k€`;
  return `${sign}${abs.toFixed(0)} €`;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function hexToRgb(hex) {
  const clean = hex.replace("#", "");
  const full = clean.length === 3
    ? clean.split("").map(ch => ch + ch).join("")
    : clean;
  const value = parseInt(full, 16);
  return {
    r: (value >> 16) & 255,
    g: (value >> 8) & 255,
    b: value & 255,
  };
}

function rgbToHex({ r, g, b }) {
  const toHex = (v) => Math.round(v).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function mixHex(a, b, t) {
  const from = hexToRgb(a);
  const to = hexToRgb(b);
  return rgbToHex({
    r: from.r + (to.r - from.r) * t,
    g: from.g + (to.g - from.g) * t,
    b: from.b + (to.b - from.b) * t,
  });
}

function accessibilityColor(value, maxAbs) {
  const bound = Math.max(Math.abs(maxAbs || 0), 1);
  const t = clamp01(Math.abs(value) / bound);
  if (value > 0) return mixHex("#F8FAFC", "#2563EB", t);
  if (value < 0) return mixHex("#F8FAFC", "#DC2626", t);
  return "#F8FAFC";
}

/* ════════════════════════════════════════════════════════════════════════
   LANDING PAGE
   ════════════════════════════════════════════════════════════════════════ */

async function initLandingPage() {
  animateCounters();

  // Charge villes + profils en parallèle
  const [cities, profiles] = await Promise.all([
    fetch(API.cities).then(r => r.json()),
    fetch(API.profiles).then(r => r.json()),
  ]);

  renderCities(cities);
  renderProfiles(profiles);
  bindParamSliders();
  bindToggles();
  bindRunButton();
}

function animateCounters() {
  document.querySelectorAll(".count-up").forEach((el) => {
    const target = parseInt(el.dataset.target, 10);
    if (isNaN(target)) return;
    const duration = 1100;
    const start = performance.now();
    function step(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);  // easeOutCubic
      el.textContent = Math.round(target * eased);
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
}

function renderCities(cities) {
  const grid = document.getElementById("cities-grid");
  if (!grid) return;
  grid.innerHTML = "";
  cities.forEach((c, i) => {
    const card = document.createElement("button");
    card.className = "city-card" + (i === 0 ? " active" : "");
    card.dataset.osm = c.osm;
    card.dataset.label = c.label;
    card.innerHTML = `
      <div class="name">${c.label}</div>
      <div class="meta">${fmtNumber(c.population)} hab. · ${c.osm.split(",")[1]?.trim() || ""}</div>
      <span class="badge">${c.size}</span>
    `;
    card.onclick = () => {
      grid.querySelectorAll(".city-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
    };
    grid.appendChild(card);
  });
}

function renderProfiles(profiles) {
  const pills = document.getElementById("profile-pills");
  if (!pills) return;
  const emoji = { ecolo: "🌳", mobilite: "🚗", economique: "💼", equilibre: "⚖️" };
  pills.innerHTML = "";
  profiles.forEach((p, i) => {
    const btn = document.createElement("button");
    btn.className = "pill" + (i === 0 ? " active" : "");
    btn.dataset.name = p.name;
    btn.innerHTML = `<span class="emoji">${emoji[p.name] || "•"}</span>${p.label}`;
    btn.onclick = () => {
      pills.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
      btn.classList.add("active");
    };
    pills.appendChild(btn);
  });
}

function bindParamSliders() {
  const bindings = [
    ["p-horizon", "v-horizon", (v) => `${v} ans`],
    ["p-hour", "v-hour", (v) => `${v} h`],
    ["p-cells", "v-cells", (v) => `${v} cellules`],
    ["p-scale", "v-scale", (v) => `${(v / 10).toFixed(1)}×`],
    ["p-fw", "v-fw", (v) => `${v} FW`],
    ["p-cand", "v-cand", (v) => `${v}`],
    ["p-periph", "v-periph", (v) => `${v} m`],
    ["p-access", "v-access", (v) => `${v} min`],
    ["p-ueiter", "v-ueiter", (v) => `${v}`],
  ];
  bindings.forEach(([slider, display, fmt]) => {
    const s = document.getElementById(slider);
    const d = document.getElementById(display);
    if (!s || !d) return;
    const update = () => d.textContent = fmt(s.value);
    s.addEventListener("input", update);
    update();
  });
}

function bindToggles() {
  [
    "toggle-multi", "toggle-robustness", "toggle-pareto", "toggle-braess",
    "toggle-simplified", "toggle-route500",
  ].forEach(id => {
    const tg = document.getElementById(id);
    if (tg) tg.addEventListener("click", () => tg.classList.toggle("on"));
  });
}

function bindRunButton() {
  const btn = document.getElementById("btn-run");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    const activeCity = document.querySelector(".city-card.active");
    const activeProfile = document.querySelector(".pill.active");
    if (!activeCity || !activeProfile) {
      alert("Sélectionnez une ville et un profil.");
      return;
    }

    // Helpers safes (null-safe : si l'élément manque, on garde la valeur par défaut)
    const slider = (id, fallback) => {
      const el = document.getElementById(id);
      if (!el) { console.warn(`[uo] slider #${id} manquant`); return fallback; }
      return el.value;
    };
    const toggle = (id) => {
      const el = document.getElementById(id);
      if (!el) { console.warn(`[uo] toggle #${id} manquant`); return false; }
      return el.classList.contains("on");
    };

    const payload = {
      city: activeCity.dataset.osm,
      profile: activeProfile.dataset.name,
      hour: parseInt(slider("p-hour", "8"), 10),
      n_cells: parseInt(slider("p-cells", "10"), 10),
      scale_factor: parseInt(slider("p-scale", "3"), 10) / 10,
      horizon_years: parseInt(slider("p-horizon", "10"), 10),
      max_candidates: parseInt(slider("p-cand", "30"), 10),
      max_fw_evals: parseInt(slider("p-fw", "10"), 10),
      periphery_margin_m: parseFloat(slider("p-periph", "600")),
      access_threshold_min: parseInt(slider("p-access", "15"), 10),
      max_iter_ue: parseInt(slider("p-ueiter", "100"), 10),
      include_route500: toggle("toggle-route500"),
      simplified_highway: toggle("toggle-simplified"),
      multi_profile: toggle("toggle-multi"),
      include_robustness: toggle("toggle-robustness"),
      include_pareto: toggle("toggle-pareto"),
      include_braess: toggle("toggle-braess"),
    };
    console.log("[uo] payload :", payload);

    btn.disabled = true;
    btn.querySelector(".arrow").textContent = "…";

    try {
      const resp = await fetch(API.createJob, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const { job_id } = await resp.json();
      sessionStorage.setItem(SS_KEY_JOB, JSON.stringify({ job_id, payload }));
      showLoader();
      pollAndRedirect(job_id);
    } catch (err) {
      alert(`Erreur au lancement : ${err.message}`);
      btn.disabled = false;
      btn.querySelector(".arrow").textContent = "→";
    }
  });
}

function showLoader() {
  const loader = document.getElementById("loader");
  if (loader) loader.classList.add("visible");
}

function hideLoader() {
  const loader = document.getElementById("loader");
  if (loader) loader.classList.remove("visible");
}

async function pollAndRedirect(jobId) {
  const pctEl = document.getElementById("loader-pct");
  const elapsedEl = document.getElementById("loader-elapsed");
  const fillEl = document.getElementById("loader-fill");
  const stepEl = document.getElementById("loader-step");

  console.log(`[uo] polling job ${jobId}`);
  let consecFails = 0;

  while (true) {
    let data;
    try {
      const resp = await fetch(API.jobStatus(jobId));
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      data = await resp.json();
      consecFails = 0;
    } catch (err) {
      consecFails += 1;
      console.warn(`[uo] poll fetch failed (${consecFails}):`, err);
      if (stepEl) stepEl.textContent = `Connexion perdue (${consecFails}) — reconnexion…`;
      if (consecFails >= 8) {
        hideLoader();
        alert(`Impossible de joindre le serveur après ${consecFails} essais.\nJob ID : ${jobId}\nVérifie que uvicorn tourne (lance \`uvicorn api.main:app --reload --port 8000\`).`);
        resetRunButton();
        return;
      }
      await new Promise(r => setTimeout(r, 2000));
      continue;
    }

    console.log(`[uo] job ${jobId} : ${data.status} / ${(data.progress * 100).toFixed(0)}% / ${data.elapsed_s}s / "${data.step}"`);
    if (pctEl) pctEl.textContent = Math.round((data.progress || 0) * 100);
    if (elapsedEl) elapsedEl.textContent = data.elapsed_s || 0;
    if (fillEl) fillEl.style.width = `${(data.progress || 0) * 100}%`;
    if (stepEl) stepEl.textContent = data.step || "…";

    if (data.status === "done") {
      console.log(`[uo] job ${jobId} done → redirect /dashboard`);
      window.location.href = `/dashboard?job=${jobId}`;
      return;
    }
    if (data.status === "error") {
      hideLoader();
      alert(`Le pipeline a échoué : ${data.error}`);
      resetRunButton();
      return;
    }
    await new Promise(r => setTimeout(r, 1500));
  }
}

function resetRunButton() {
  const btn = document.getElementById("btn-run");
  if (btn) {
    btn.disabled = false;
    const arrow = btn.querySelector(".arrow");
    if (arrow) arrow.textContent = "→";
  }
}

/* ════════════════════════════════════════════════════════════════════════
   DASHBOARD PAGE
   ════════════════════════════════════════════════════════════════════════ */

async function initDashboardPage() {
  const params = new URLSearchParams(window.location.search);
  let jobId = params.get("job");
  if (!jobId) {
    const last = sessionStorage.getItem(SS_KEY_JOB);
    if (last) {
      try { jobId = JSON.parse(last).job_id; } catch {}
    }
  }
  if (!jobId) {
    showEmptyState();
    return;
  }

  const resp = await fetch(API.jobStatus(jobId));
  const data = await resp.json();
  if (data.status !== "done") {
    showEmptyState(`Job ${jobId} : status=${data.status}`);
    return;
  }

  document.getElementById("dashboard").style.display = "";
  document.getElementById("empty").style.display = "none";
  renderDashboard(data);
}

function showEmptyState(msg) {
  document.getElementById("dashboard").style.display = "none";
  document.getElementById("empty").style.display = "";
  if (msg) {
    const m = document.querySelector("#empty p");
    if (m) m.textContent = msg;
  }
}

function renderDashboard(data) {
  // Header
  document.getElementById("d-city").textContent = data.city.split(",")[0];
  const profiles = data.profiles || [];
  const main = profiles[0];
  if (main) {
    document.getElementById("d-profile").textContent = main.profile_label;
  }
  // Budget + horizon (calculés automatiquement)
  const bdg = data.budget;
  if (bdg) {
    const src = bdg.source === "OFGL" ? "OFGL" : "estimé pop";
    document.getElementById("d-budget").textContent =
      `${(bdg.total_eur / 1e6).toFixed(1)} M€ (${src}, ${data.horizon_years} ans)`;
  } else {
    document.getElementById("d-budget").textContent = "—";
  }

  // Synthèse long-terme
  renderForecastFrame(data);

  // KPIs principaux (effet du plan)
  renderKpis(main.kpis);

  // Carte (avec Braess geojson optionnel)
  initMap(data.network_geojson, data.plan_geojson, data.braess_geojson, data.accessibility_geojson);

  // Liste interventions
  renderInterventions(main.interventions);

  // Impact spatial par zone
  renderAccessibilityImpact(data.accessibility_geojson);

  // Mode comparaison ?
  if (profiles.length > 1) {
    renderCompare(profiles);
    document.getElementById("compare-block").classList.remove("hidden");
  }

  // Robustesse
  if (data.robustness) {
    renderRobustness(data.robustness);
    document.getElementById("robust-block").classList.remove("hidden");
  }

  // Pareto
  if (data.pareto) {
    renderPareto(data.pareto);
    document.getElementById("pareto-block").classList.remove("hidden");
  }

  // Braess
  if (data.braess && data.braess.length > 0) {
    renderBraess(data.braess);
    document.getElementById("braess-block").classList.remove("hidden");
  }

  // Footer technique
  if (data.baseline) {
    renderTech(data.baseline);
  }

  // Toggle avant/après
  bindMapToggle();
}

/* ─── Cadre long-terme : horizon + projection ML + budget OFGL ──────── */

function renderForecastFrame(data) {
  const block = document.getElementById("forecast-block");
  const container = document.getElementById("kpis-forecast");
  if (!container || !block) return;
  container.innerHTML = "";
  const cards = [];
  const horizon = data.horizon_years || 0;
  cards.push({
    label: "Horizon",
    value: `${horizon} ans`,
    delta: "planification retenue",
    accent: "neutral",
  });
  const bdg = data.budget;
  if (bdg) {
    const sourceLabel = bdg.source === "OFGL"
      ? `OFGL · ${bdg.annual_voirie_eur ? (bdg.annual_voirie_eur / 1e6).toFixed(1) + " M€/an voirie" : ""}`
      : `estimé pop · ${bdg.annual_voirie_eur ? (bdg.annual_voirie_eur / 1e6).toFixed(1) + " M€/an" : ""}`;
    cards.push({
      label: "Budget travaux cumulé",
      value: fmtEur(bdg.total_eur),
      delta: sourceLabel,
      accent: "good",
    });
  }
  if (cards.length === 0) {
    block.classList.add("hidden");
    return;
  }
  block.classList.remove("hidden");
  cards.forEach(k => {
    const div = document.createElement("div");
    div.className = `kpi ${k.accent || "neutral"}`;
    div.innerHTML = `
      <div class="label">${k.label}</div>
      <div class="value">${k.value}</div>
      ${k.delta ? `<div class="delta">${k.delta}</div>` : ""}
    `;
    container.appendChild(div);
  });
}

/* ─── Braess ─────────────────────────────────────────────────────────── */

function renderBraess(items) {
  const body = document.getElementById("braess-body");
  body.innerHTML = "";
  items.forEach(b => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>#${b.rank}</strong></td>
      <td class="num">${b.edge_id}</td>
      <td>${b.highway}</td>
      <td class="num">${b.length_m.toFixed(0)} m</td>
      <td class="num">${b.delta_vht_h >= 0 ? "+" : ""}${fmtNumber(b.delta_vht_h, { maximumFractionDigits: 1 })} h</td>
      <td class="num">${fmtEur(b.annual_benefit_eur)}/an</td>
    `;
    body.appendChild(tr);
  });
}

/* ─── Footer technique ───────────────────────────────────────────────── */

function renderTech(b) {
  const txt = document.getElementById("tech-text");
  if (!txt) return;
  txt.textContent = [
    `Réseau            : ${fmtNumber(b.n_nodes)} nœuds, ${fmtNumber(b.n_edges)} arcs`,
    `Demande           : ${b.n_zones} zones, ${fmtNumber(b.total_trips, { maximumFractionDigits: 0 })} véh/h`,
    `VHT (h pointe)    : ${fmtNumber(b.vht_h, { maximumFractionDigits: 1 })}`,
    `Surcoût congestion: +${b.congestion_overhead_pct.toFixed(1)}%`,
    `Arcs saturés ≥0.9 : ${b.n_saturated_arcs} / ${fmtNumber(b.n_edges)}`,
    `FW gap            : ${b.fw_gap.toExponential(1)} (${b.fw_iterations} iter)`,
  ].join("\n");
}

/* ─── Robustesse ─────────────────────────────────────────────────────── */

function renderRobustness(rob) {
  const verdict = document.getElementById("robust-verdict");
  const icon = document.getElementById("robust-icon");
  const label = document.getElementById("robust-label");
  const sub = document.getElementById("robust-sub");

  verdict.classList.remove("good", "bad");
  if (rob.is_robust) {
    verdict.classList.add("good");
    icon.textContent = "✓";
    label.textContent = "Plan robuste";
    sub.textContent = `Bénéfique sur les 4 scénarios. Pire cas : demande ×${rob.worst_scale.toFixed(2)} (+${fmtEur(rob.worst_benefit_eur)}/an).`;
  } else {
    verdict.classList.add("bad");
    icon.textContent = "✗";
    label.textContent = "Plan fragile";
    sub.textContent = `Déficitaire dans ${rob.n_failing}/${rob.points.length} scénario(s). Pire cas : demande ×${rob.worst_scale.toFixed(2)} (${fmtEur(rob.worst_benefit_eur)}/an).`;
  }

  const body = document.getElementById("robust-body");
  body.innerHTML = "";
  rob.points.forEach(p => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>×${p.demand_scale.toFixed(2)}</strong></td>
      <td class="num">${fmtNumber(p.baseline_vht_h, { maximumFractionDigits: 0 })} h</td>
      <td class="num">${fmtNumber(p.plan_vht_h, { maximumFractionDigits: 0 })} h</td>
      <td class="num">${p.delta_vht_h >= 0 ? "+" : ""}${fmtNumber(p.delta_vht_h, { maximumFractionDigits: 1 })} h</td>
      <td class="num">${p.annual_benefit_eur >= 0 ? "+" : ""}${fmtEur(p.annual_benefit_eur)}/an</td>
      <td><span class="status-tag ${p.is_beneficial ? "good" : "bad"}">${p.is_beneficial ? "✓ rentable" : "✗ déficit"}</span></td>
    `;
    body.appendChild(tr);
  });
}

/* ─── Pareto ─────────────────────────────────────────────────────────── */

let PARETO_CHART = null;
let IMPACT_CHART = null;

function renderPareto(par) {
  const body = document.getElementById("pareto-body");
  body.innerHTML = "";
  par.points.forEach(p => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${(p.budget_eur / 1e6).toFixed(0)} M€</strong></td>
      <td class="num">${(p.capex_used_eur / 1e6).toFixed(1)} M€</td>
      <td class="num">${p.n_interventions}</td>
      <td class="num">${p.joint_annual_benefit_eur >= 0 ? "+" : ""}${fmtEur(p.joint_annual_benefit_eur)}</td>
      <td class="num">${p.delta_vht_h >= 0 ? "+" : ""}${fmtNumber(p.delta_vht_h, { maximumFractionDigits: 0 })} h</td>
      <td class="num">${p.bcr.toFixed(2)}</td>
    `;
    body.appendChild(tr);
  });

  // Sweet spot
  const sweet = document.getElementById("sweet-spot");
  if (par.sweet_spot_budget_eur != null) {
    sweet.classList.remove("hidden");
    document.getElementById("sweet-spot-value").textContent =
      `~${(par.sweet_spot_budget_eur / 1e6).toFixed(0)} M€ → ${fmtEur(par.sweet_spot_benefit_eur)}/an (${fmtNumber(par.sweet_spot_marginal, { maximumFractionDigits: 0 })} €/an par M€ investi)`;
  } else {
    sweet.classList.add("hidden");
  }

  // Chart.js : CAPEX utilisé (x) vs bénéfice annuel (y)
  const ctx = document.getElementById("pareto-chart");
  if (!ctx || typeof Chart === "undefined") return;
  if (PARETO_CHART) PARETO_CHART.destroy();

  const dataPoints = par.points.map(p => ({
    x: p.capex_used_eur / 1e6,
    y: p.joint_annual_benefit_eur / 1e6,
  }));

  PARETO_CHART = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [{
        label: "Bénéfice annuel (M€/an)",
        data: dataPoints,
        borderColor: "#FF5C39",
        backgroundColor: "rgba(255, 92, 57, 0.12)",
        borderWidth: 3,
        tension: 0.35,
        pointRadius: 6,
        pointHoverRadius: 9,
        pointBackgroundColor: "#FF5C39",
        pointBorderColor: "#FFFFFF",
        pointBorderWidth: 2,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0E1116",
          padding: 12,
          titleFont: { family: "Inter", weight: "600" },
          bodyFont: { family: "JetBrains Mono", size: 12 },
          callbacks: {
            title: (ctx) => `CAPEX ${ctx[0].parsed.x.toFixed(1)} M€`,
            label: (ctx) => `Bénéfice : +${ctx.parsed.y.toFixed(2)} M€/an`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "CAPEX utilisé (M€)", font: { family: "Inter", weight: "600", size: 12 } },
          grid: { color: "#F0F0EC" },
          ticks: { font: { family: "JetBrains Mono", size: 11 }, color: "#6B7280" },
        },
        y: {
          title: { display: true, text: "Bénéfice annuel (M€/an)", font: { family: "Inter", weight: "600", size: 12 } },
          grid: { color: "#F0F0EC" },
          ticks: { font: { family: "JetBrains Mono", size: 11 }, color: "#6B7280" },
        },
      },
    },
  });
}

function median(values) {
  if (!values || values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2) return sorted[mid];
  return (sorted[mid - 1] + sorted[mid]) / 2;
}

function renderAccessibilityImpact(accessGeo) {
  const block = document.getElementById("impact-block");
  const summary = document.getElementById("impact-summary");
  const body = document.getElementById("impact-body");
  const canvas = document.getElementById("impact-chart");
  if (!block || !summary || !body || !canvas) return;

  if (!accessGeo || !accessGeo.features || !accessGeo.features.length) {
    block.classList.add("hidden");
    return;
  }

  const rows = accessGeo.features.map(f => ({
    zone_id: String(f.properties?.zone_id ?? "zone"),
    before: Number(f.properties?.access_before ?? 0),
    after: Number(f.properties?.access_after ?? 0),
    delta: Number(f.properties?.access_delta ?? 0),
    pct: f.properties?.access_delta_pct,
    population: Number(f.properties?.population ?? 0),
    jobs: Number(f.properties?.jobs ?? 0),
  })).filter(r => Number.isFinite(r.before) && Number.isFinite(r.after) && Number.isFinite(r.delta));

  if (!rows.length) {
    block.classList.add("hidden");
    return;
  }

  const gains = rows.filter(r => r.delta > 0);
  const losses = rows.filter(r => r.delta < 0);
  const avgDelta = rows.reduce((sum, r) => sum + r.delta, 0) / rows.length;
  const medDelta = median(rows.map(r => r.delta));
  const best = rows.reduce((acc, r) => (r.delta > acc.delta ? r : acc), rows[0]);
  const worst = rows.reduce((acc, r) => (r.delta < acc.delta ? r : acc), rows[0]);

  summary.innerHTML = [
    {
      label: "Zones en hausse",
      value: `${gains.length}/${rows.length}`,
      accent: "good",
    },
    {
      label: "Δ moyen",
      value: `${avgDelta >= 0 ? "+" : ""}${avgDelta.toFixed(1)} zones`,
      accent: avgDelta >= 0 ? "good" : "bad",
    },
    {
      label: "Δ médian",
      value: `${medDelta >= 0 ? "+" : ""}${medDelta.toFixed(1)} zones`,
      accent: medDelta >= 0 ? "good" : "bad",
    },
    {
      label: "Meilleur gain",
      value: `${best.zone_id} · +${best.delta.toFixed(1)}`,
      accent: "good",
    },
    {
      label: "Pire perte",
      value: `${worst.zone_id} · ${worst.delta >= 0 ? "+" : ""}${worst.delta.toFixed(1)}`,
      accent: worst.delta >= 0 ? "good" : "bad",
    },
  ].map(stat => `
    <div class="impact-stat ${stat.accent || "neutral"}">
      <div class="label">${stat.label}</div>
      <div class="value">${stat.value}</div>
    </div>
  `).join("");

  const ranked = [...rows].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 12);
  body.innerHTML = "";
  ranked.forEach(r => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${r.zone_id}</strong></td>
      <td class="num">${fmtNumber(r.before, { maximumFractionDigits: 1 })}</td>
      <td class="num">${fmtNumber(r.after, { maximumFractionDigits: 1 })}</td>
      <td class="num">${r.delta >= 0 ? "+" : ""}${fmtNumber(r.delta, { maximumFractionDigits: 1 })}</td>
    `;
    body.appendChild(tr);
  });

  block.classList.remove("hidden");

  const chartRows = [...rows].sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta)).slice(0, 15);
  const labels = chartRows.map(r => r.zone_id);
  const values = chartRows.map(r => r.delta);
  const colors = chartRows.map(r => (r.delta >= 0 ? "rgba(37, 99, 235, 0.72)" : "rgba(220, 38, 38, 0.72)"));
  const maxAbs = Math.max(1, ...values.map(v => Math.abs(v)));

  try {
    if (IMPACT_CHART) IMPACT_CHART.destroy();
    IMPACT_CHART = new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Δ accessibilité (zones)",
          data: values,
          backgroundColor: colors,
          borderColor: colors,
          borderWidth: 1,
          borderRadius: 8,
          maxBarThickness: 20,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#0E1116",
            padding: 12,
            titleFont: { family: "Inter", weight: "600" },
            bodyFont: { family: "JetBrains Mono", size: 12 },
            callbacks: {
              title: (items) => items[0].label,
              label: (ctx) => {
                const value = Number(ctx.raw ?? 0);
                return `Δ : ${value >= 0 ? "+" : ""}${value.toFixed(1)} zones`;
              },
            },
          },
        },
        scales: {
          x: {
            min: -maxAbs * 1.1,
            max: maxAbs * 1.1,
            grid: { color: "#F0F0EC" },
            ticks: { font: { family: "JetBrains Mono", size: 11 }, color: "#6B7280" },
            title: { display: true, text: "Δ accessibilité (zones joignables)", font: { family: "Inter", weight: "600", size: 12 } },
          },
          y: {
            grid: { display: false },
            ticks: { font: { family: "JetBrains Mono", size: 11 }, color: "#6B7280" },
          },
        },
      },
    });
  } catch (err) {
    console.warn("[uo] impact chart failed:", err);
    canvas.insertAdjacentHTML("afterend", "<p class=\"muted\" style=\"padding: 12px 0 0 16px;\">Le graphique n’a pas pu être dessiné, mais le tableau des zones reste disponible.</p>");
  }
}

function renderKpis(kpis) {
  const container = document.getElementById("kpis");
  container.innerHTML = "";
  (kpis || []).forEach(k => {
    const div = document.createElement("div");
    div.className = `kpi ${k.accent || "neutral"}`;
    div.innerHTML = `
      <div class="label">${k.label}</div>
      <div class="value">${k.value}</div>
      ${k.delta ? `<div class="delta">${k.delta}</div>` : ""}
    `;
    container.appendChild(div);
  });
}

function renderInterventions(items) {
  const list = document.getElementById("interventions-list");
  list.innerHTML = "";
  if (!items || items.length === 0) {
    list.innerHTML = '<p class="intervention-empty">Aucune intervention rentable identifiée sous ce budget.</p>';
    return;
  }
  items.forEach(it => {
    const div = document.createElement("div");
    div.className = "intervention";
    // Le backend fournit it.type ("corridor" | "upgrade" | "new_route")
    div.dataset.type = it.type || "corridor";
    const detourStr = it.detour_before ? `${it.detour_before.toFixed(2)}×` : "—";
    div.innerHTML = `
      <div>
        <span class="rank">#${it.rank}</span>
        <span class="action">${it.action}</span>
        <span class="intervention-meta">${it.highway} · ${it.length_m.toFixed(0)} m</span>
      </div>
      <div class="row"><span>Détour avant</span><span class="v">${detourStr}</span></div>
      <div class="row"><span>ΔVHT pointe</span><span class="v">${it.delta_vht_h >= 0 ? "+" : ""}${fmtNumber(it.delta_vht_h, { maximumFractionDigits: 1 })} h</span></div>
      <div class="row"><span>Bénéfice annuel</span><span class="v">${fmtEur(it.annual_benefit_eur)}/an</span></div>
      <div class="row"><span>Coût construction</span><span class="v">${fmtEur(it.construction_cost_eur)}</span></div>
      <div class="row"><span>BCR · payback</span><span class="v">${it.bcr.toFixed(2)} · ${it.payback_years !== null ? it.payback_years.toFixed(1) + " ans" : "∞"}</span></div>
    `;
    list.appendChild(div);
  });
}

function renderCompare(profiles) {
  const head = document.getElementById("compare-header");
  const body = document.getElementById("compare-body");
  head.innerHTML = "<th>Indicateur</th>" + profiles.map(p => `<th><span class="profile-tag ${p.profile_name}">${p.profile_label}</span></th>`).join("");

  const rows = [
    ["VHT après", p => fmtNumber(p.joint_vht_h, { maximumFractionDigits: 0 }) + " h"],
    ["Score annuel après", p => fmtEur(p.joint_score_eur)],
    ["Gain annuel", p => fmtEur(p.baseline_score_eur - p.joint_score_eur)],
    ["Interventions", p => `${p.interventions.length}`],
    ["CAPEX cumulé", p => fmtEur(p.interventions.reduce((s, i) => s + i.construction_cost_eur, 0))],
    ["Accessibilité (zones)", p => `${p.accessibility_before.toFixed(1)} → ${p.accessibility_after.toFixed(1)}`],
    ["Gini équité", p => `${p.gini_before.toFixed(3)} → ${p.gini_after.toFixed(3)}`],
  ];

  body.innerHTML = "";
  rows.forEach(([label, fn]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><strong>${label}</strong></td>` + profiles.map(p => `<td class="num">${fn(p)}</td>`).join("");
    body.appendChild(tr);
  });
}

/* ─── Carte Leaflet ──────────────────────────────────────────────────── */

let MAP = null;
let LAYER_BEFORE = null;
let LAYER_AFTER = null;
let LAYER_PLAN = null;
let LAYER_BRAESS = null;
let LAYER_ACCESS = null;
let LAYER_HEAT_GAIN = null;
let LAYER_HEAT_LOSS = null;
let CURRENT_MODE = "after";
let NETWORK_FEATURE_COUNT = 0;

function satStyle(s) {
  if (s < 0.5) return { color: "#9EE493", weight: 1.5, opacity: 0.55 };
  if (s < 0.8) return { color: "#FFC857", weight: 2.5, opacity: 0.85 };
  if (s < 0.95) return { color: "#F47C3C", weight: 3.5, opacity: 0.95 };
  return { color: "#C92020", weight: 4.5, opacity: 1.0 };
}

const PLAN_STYLES = {
  corridor: { color: "#2563EB", weight: 6, opacity: 0.9 },
  upgrade: { color: "#DB2777", weight: 6, opacity: 0.9 },
  new_route: { color: "#16A34A", weight: 6, opacity: 0.9 },
};

function initMap(networkGeo, planGeo, braessGeo, accessibilityGeo) {
  if (!networkGeo || !networkGeo.features.length) {
    document.getElementById("map").innerHTML = "<p style='padding:20px; color:var(--muted);'>Aucun arc à afficher.</p>";
    return;
  }
  NETWORK_FEATURE_COUNT = networkGeo.features.length;

  // Centre depuis bbox
  let minLat = 90, maxLat = -90, minLng = 180, maxLng = -180;
  networkGeo.features.forEach(f => {
    f.geometry.coordinates.forEach(([x, y]) => {
      if (y < minLat) minLat = y;
      if (y > maxLat) maxLat = y;
      if (x < minLng) minLng = x;
      if (x > maxLng) maxLng = x;
    });
  });
  const center = [(minLat + maxLat) / 2, (minLng + maxLng) / 2];

  MAP = L.map("map", { zoomControl: true, attributionControl: false }).setView(center, 13);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
  }).addTo(MAP);

  LAYER_BEFORE = L.layerGroup();
  LAYER_AFTER = L.layerGroup();
  networkGeo.features.forEach(f => {
    const coordsLatLng = f.geometry.coordinates.map(([x, y]) => [y, x]);
    L.polyline(coordsLatLng, satStyle(f.properties.sat_before)).addTo(LAYER_BEFORE);
    L.polyline(coordsLatLng, satStyle(f.properties.sat_after)).addTo(LAYER_AFTER);
  });

  // Plan overlay
  LAYER_PLAN = L.layerGroup();
  if (planGeo) {
    planGeo.features.forEach(f => {
      const coordsLatLng = f.geometry.coordinates.map(([x, y]) => [y, x]);
      const style = PLAN_STYLES[f.properties.type] || PLAN_STYLES.corridor;
      const line = L.polyline(coordsLatLng, style);
      line.bindTooltip(
        `<strong>#${f.properties.rank} — ${f.properties.action}</strong><br>` +
        `${f.properties.highway} · ${f.properties.length_m.toFixed(0)} m<br>` +
        `Bénéfice : ${fmtEur(f.properties.annual_benefit_eur)}/an<br>` +
        `Coût : ${fmtEur(f.properties.construction_cost_eur)}<br>` +
        `BCR : ${f.properties.bcr.toFixed(2)}`,
        { sticky: true }
      );
      line.addTo(LAYER_PLAN);

      // Marker rang au centre
      const midIdx = Math.floor(coordsLatLng.length / 2);
      L.marker(coordsLatLng[midIdx], {
        icon: L.divIcon({
          className: "plan-marker",
          html: `<div style="background: ${style.color}; color: white; border-radius: 50%; width: 28px; height: 28px; display: grid; place-items: center; font-weight: 800; font-family: 'Space Grotesk', sans-serif; box-shadow: 0 2px 8px rgba(0,0,0,0.25); border: 2px solid white;">${f.properties.rank}</div>`,
          iconSize: [28, 28], iconAnchor: [14, 14],
        }),
      }).addTo(LAYER_PLAN);
    });
  }

  // Braess overlay (à supprimer)
  LAYER_BRAESS = L.layerGroup();
  if (braessGeo && braessGeo.features) {
    braessGeo.features.forEach(f => {
      const coordsLatLng = f.geometry.coordinates.map(([x, y]) => [y, x]);
      const line = L.polyline(coordsLatLng, {
        color: "#D22B2B",
        weight: 6,
        opacity: 0.95,
        dashArray: "10, 8",
      });
      line.bindTooltip(
        `<strong>SUPPRESSION #${f.properties.rank}</strong><br>` +
        `${f.properties.highway} · ${f.properties.length_m.toFixed(0)} m<br>` +
        `ΔVHT en retirant : ${f.properties.delta_vht_h >= 0 ? "+" : ""}${f.properties.delta_vht_h.toFixed(1)} h<br>` +
        `Économie : ${fmtEur(f.properties.annual_benefit_eur)}/an`,
        { sticky: true }
      );
      line.addTo(LAYER_BRAESS);
    });
  }

  // Heatmap d'impact spatial (vraie heatmap, centrée sur les zones)
  LAYER_HEAT_GAIN = null;
  LAYER_HEAT_LOSS = null;
  if (accessibilityGeo && accessibilityGeo.features && accessibilityGeo.features.length && typeof L.heatLayer === "function") {
    const gainPoints = [];
    const lossPoints = [];
    const deltas = accessibilityGeo.features
      .map(f => Number(f.properties?.access_delta ?? 0))
      .filter(v => Number.isFinite(v));
    const maxAbs = Math.max(1, ...deltas.map(v => Math.abs(v)));

    accessibilityGeo.features.forEach(f => {
      const p = f.properties || {};
      const centroid = Array.isArray(p.centroid) && p.centroid.length >= 2 ? p.centroid : null;
      if (!centroid) return;
      const lng = Number(centroid[0]);
      const lat = Number(centroid[1]);
      const delta = Number(p.access_delta ?? 0);
      if (!Number.isFinite(lat) || !Number.isFinite(lng) || !Number.isFinite(delta) || delta === 0) return;
      const intensity = Math.max(0.15, Math.min(1.0, Math.abs(delta) / maxAbs));
      const point = [lat, lng, intensity];
      if (delta > 0) gainPoints.push(point);
      else lossPoints.push(point);
    });

    if (gainPoints.length) {
      LAYER_HEAT_GAIN = L.heatLayer(gainPoints, {
        radius: 42,
        blur: 28,
        maxZoom: 17,
        minOpacity: 0.30,
        gradient: {
          0.2: "#DBEAFE",
          0.45: "#60A5FA",
          0.7: "#2563EB",
          1.0: "#0F172A",
        },
      });
    }
    if (lossPoints.length) {
      LAYER_HEAT_LOSS = L.heatLayer(lossPoints, {
        radius: 42,
        blur: 28,
        maxZoom: 17,
        minOpacity: 0.30,
        gradient: {
          0.2: "#FEE2E2",
          0.45: "#F87171",
          0.7: "#DC2626",
          1.0: "#7F1D1D",
        },
      });
    }
  } else if (accessibilityGeo && accessibilityGeo.features && accessibilityGeo.features.length) {
    console.warn("[uo] leaflet.heat non disponible, impact spatial en fallback simple.");
  }

  // Mode initial : "après plan"
  LAYER_AFTER.addTo(MAP);
  LAYER_PLAN.addTo(MAP);
  LAYER_BRAESS.addTo(MAP);

  document.getElementById("map-info").textContent = `${NETWORK_FEATURE_COUNT} arcs affichés`;
}

function bindMapToggle() {
  const buttons = document.querySelectorAll("#map-toggle button");
  buttons.forEach(b => {
    b.addEventListener("click", () => {
      buttons.forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      switchMapMode(b.dataset.mode);
    });
  });
}

function switchMapMode(mode) {
  if (!MAP || !LAYER_BEFORE || !LAYER_AFTER) return;
  // Pour la version "split" on garde l'after avec overlay pour l'instant ;
  // le vrai split-screen viendra dans la phase 2.
  MAP.removeLayer(LAYER_BEFORE);
  MAP.removeLayer(LAYER_AFTER);
  if (LAYER_PLAN) MAP.removeLayer(LAYER_PLAN);
  if (LAYER_BRAESS) MAP.removeLayer(LAYER_BRAESS);
  if (LAYER_HEAT_GAIN) MAP.removeLayer(LAYER_HEAT_GAIN);
  if (LAYER_HEAT_LOSS) MAP.removeLayer(LAYER_HEAT_LOSS);
  if (mode === "before") {
    LAYER_BEFORE.addTo(MAP);
  } else if (mode === "after") {
    LAYER_AFTER.addTo(MAP);
    if (LAYER_PLAN) LAYER_PLAN.addTo(MAP);
    if (LAYER_BRAESS) LAYER_BRAESS.addTo(MAP);
  } else if (mode === "impact") {
    if (LAYER_HEAT_LOSS) LAYER_HEAT_LOSS.addTo(MAP);
    if (LAYER_HEAT_GAIN) LAYER_HEAT_GAIN.addTo(MAP);
    if (LAYER_PLAN) LAYER_PLAN.addTo(MAP);
  } else {
    // split : on superpose les 2 avec opacité réduite sur le before
    LAYER_BEFORE.addTo(MAP);
    LAYER_AFTER.addTo(MAP);
    if (LAYER_PLAN) LAYER_PLAN.addTo(MAP);
    if (LAYER_BRAESS) LAYER_BRAESS.addTo(MAP);
  }
  CURRENT_MODE = mode;

  const info = document.getElementById("map-info");
  if (info) {
    if (mode === "impact") {
      info.textContent = "Heatmap : bleu = gain d'accès, rouge = perte";
    } else {
      info.textContent = `${NETWORK_FEATURE_COUNT} arcs affichés`;
    }
  }
}
