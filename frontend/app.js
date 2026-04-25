const form = document.getElementById('analyticsForm');
const insightsEl = document.getElementById('insights');
const gateStatus = document.getElementById('gateStatus');
const scenarioButton = document.getElementById('scenarioButton');
const fromSourceClean = document.getElementById('fromSourceClean');
const fromSourceMessy = document.getElementById('fromSourceMessy');

let lastPortfolioValue = 1000000;

/* ── Results heading visibility ── */

const resultsHeading = document.getElementById('resultsHeading');

new MutationObserver(() => {
  resultsHeading.classList.toggle('visible', insightsEl.hasChildNodes());
}).observe(insightsEl, { childList: true });

/* ── JSON panel toggle ── */

const jsonToggle = document.getElementById('jsonToggle');
const jsonPanel = document.getElementById('jsonPanel');

jsonToggle.addEventListener('click', () => {
  const hidden = jsonPanel.style.display === 'none';
  jsonPanel.style.display = hidden ? 'grid' : 'none';
  jsonToggle.textContent = hidden ? 'Manual Input ▾' : 'Manual Input ▸';
});

/* ── DOM helper ── */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ── Evidence badges — neutral palette (blue / grey / amber) ── */

const EVIDENCE_COLORS = {
  high: 'evidence-high',
  medium: 'evidence-medium',
  low: 'evidence-low',
  complete: 'evidence-high',
  partial: 'evidence-medium',
  minimal: 'evidence-low',
};

function evidenceBadge(level) {
  if (!level) return el('span', '', '—');
  const badge = el('span', 'evidence-badge', level);
  badge.classList.add(EVIDENCE_COLORS[level.toLowerCase()] || 'evidence-medium');
  return badge;
}

function renderEvidenceRow(payload) {
  const row = el('div', 'evidence-row');
  row.appendChild(el('span', 'evidence-label', 'Evidence: '));
  row.appendChild(evidenceBadge(payload.evidence_strength));
  row.appendChild(el('span', 'evidence-sep', ' · '));
  row.appendChild(el('span', 'evidence-label', 'Data completeness: '));
  row.appendChild(evidenceBadge(payload.data_completeness));
  return row;
}

/* ── Value formatting ── */

const WEIGHT_KEYS = /^(weight|exposure|concentration)$/i;
const RATIO_KEYS = /^(.*ratio.*|.*ter.*)$/i;
const DRAG_UNITS = new Set(['drag_factor']);

function fmtVal(value, key, units) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'number') {
    /* Explicit units override key-based guessing */
    if (units && DRAG_UNITS.has(units)) {
      return value + '× (drag factor)';
    }
    if (key && WEIGHT_KEYS.test(key) && value >= 0 && value <= 1) {
      return (value * 100).toFixed(1) + '%';
    }
    if (key && RATIO_KEYS.test(key) && value >= 0 && value <= 1) {
      return (value * 100).toFixed(2) + '%';
    }
    return String(value);
  }
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  return String(value);
}

/* ── Structured data rendering ── */

const SKIP_KEYS = new Set([
  'source', 'aggregated_limitations', 'aggregated_limitations_count',
]);

function renderObjectFlat(obj) {
  const wrap = el('div', 'kv-object');
  for (const [k, v] of Object.entries(obj)) {
    if (SKIP_KEYS.has(k)) continue;
    const row = el('div', 'kv-item');
    row.appendChild(el('span', 'kv-item-key', k.replace(/_/g, ' ') + ': '));
    if (Array.isArray(v)) {
      row.appendChild(el('span', 'kv-item-value',
        v.map((x) => fmtVal(x, k)).join(' – ')));
    } else if (v && typeof v === 'object') {
      const parts = Object.entries(v)
        .filter(([sk]) => !SKIP_KEYS.has(sk))
        .map(([sk, sv]) => sk.replace(/_/g, ' ') + ': ' + fmtVal(sv, sk));
      row.appendChild(el('span', 'kv-item-value', parts.join(', ')));
    } else {
      row.appendChild(el('span', 'kv-item-value', fmtVal(v, k)));
    }
    wrap.appendChild(row);
  }
  return wrap;
}

function renderKvRow(key, value) {
  const row = el('div', 'kv-row');
  row.appendChild(el('span', 'kv-key', key.replace(/_/g, ' ')));
  if (Array.isArray(value)) {
    if (value.length === 0) {
      row.appendChild(el('span', 'kv-val', 'None'));
    } else if (typeof value[0] === 'object') {
      const list = el('div', 'kv-nested');
      value.forEach((item) => list.appendChild(renderObjectFlat(item)));
      row.appendChild(list);
    } else {
      row.appendChild(el('span', 'kv-val',
        value.map((x) => fmtVal(x, key)).join(', ')));
    }
  } else if (value && typeof value === 'object') {
    row.appendChild(renderObjectFlat(value));
  } else {
    row.appendChild(el('span', 'kv-val', fmtVal(value, key)));
  }
  return row;
}

function renderSupportingData(data) {
  if (!data || typeof data !== 'object') return null;
  const section = el('div', 'supporting-data');
  for (const [k, v] of Object.entries(data)) {
    if (SKIP_KEYS.has(k)) continue;
    section.appendChild(renderKvRow(k, v));
  }
  return section;
}

/* ── Limitations — always visible, never collapsed ── */

const LIM_INLINE_MAX = 3;

function renderLimitations(items, heading, opts) {
  const allowEmpty = opts && opts.allowEmpty;
  if (!items || !items.length) {
    if (!allowEmpty) return null;
    const section = el('div', 'limitations-section lim-empty');
    section.appendChild(el('h4', 'lim-heading', heading || 'Method notes'));
    section.appendChild(el('p', 'lim-none', 'No limitations reported'));
    return section;
  }
  const section = el('div', 'limitations-section');
  section.appendChild(el('h4', 'lim-heading', heading || 'Method notes'));
  const list = el('ul', 'lim-list');
  const showAll = items.length <= LIM_INLINE_MAX;
  items.forEach((lim, i) => {
    const li = el('li', null, lim);
    if (!showAll && i >= LIM_INLINE_MAX) li.classList.add('lim-hidden');
    list.appendChild(li);
  });
  section.appendChild(list);
  if (!showAll) {
    const label = (heading || 'limitations').toLowerCase();
    const toggle = el('button', 'lim-toggle',
      'Show all ' + items.length + ' ' + label);
    toggle.type = 'button';
    toggle.addEventListener('click', () => {
      list.querySelectorAll('.lim-hidden').forEach(
        (n) => n.classList.remove('lim-hidden'));
      toggle.remove();
    });
    section.appendChild(toggle);
  }
  return section;
}

function renderUnavailable(items) {
  if (!items || !items.length) return null;
  const section = el('div', 'unavailable-section');
  section.appendChild(el('h4', 'lim-heading', 'Unavailable components'));
  const list = el('ul', 'lim-list');
  items.forEach((c) => list.appendChild(el('li', null, c.replace(/_/g, ' '))));
  section.appendChild(list);
  return section;
}

/* ── Template-specific content ── */

function bodyDiagnostic(p) {
  const c = el('div', 'insight-body');
  if (p.observation) c.appendChild(el('p', 'obs', p.observation));
  if (p.why_it_matters) c.appendChild(el('p', 'context', p.why_it_matters));
  const sd = renderSupportingData(p.supporting_data);
  if (sd) {
    c.appendChild(el('h4', 'sec-heading', 'Supporting data'));
    c.appendChild(sd);
  }
  const agg = p.supporting_data?.aggregated_limitations;
  c.appendChild(
    renderLimitations(
      agg || [],
      'Underlying data limitations (aggregated across funds)',
      { allowEmpty: true },
    )
  );
  return c;
}

function bodyBenchmark(p) {
  const c = el('div', 'insight-body');
  if (p.observation) c.appendChild(el('p', 'obs', p.observation));
  if (p.benchmark) {
    c.appendChild(el('h4', 'sec-heading', 'Benchmark'));
    c.appendChild(renderObjectFlat(p.benchmark));
  }
  const sd = renderSupportingData(p.supporting_data);
  if (sd) {
    c.appendChild(el('h4', 'sec-heading', 'Supporting data'));
    c.appendChild(sd);
  }
  return c;
}

function bodyScenario(p) {
  const c = el('div', 'insight-body');
  if (p.scenario_definition) {
    c.appendChild(el('h4', 'sec-heading', 'Scenario definition'));
    c.appendChild(renderObjectFlat(p.scenario_definition));
  }
  if (p.assumptions) {
    c.appendChild(el('h4', 'sec-heading', 'Assumptions'));
    c.appendChild(renderObjectFlat(p.assumptions));
  }
  if (p.projected_impact) {
    c.appendChild(el('h4', 'sec-heading', 'Projected impact'));
    c.appendChild(renderObjectFlat(p.projected_impact));
  }
  if (p.sensitivity && p.sensitivity.length) {
    c.appendChild(el('h4', 'sec-heading', 'Sensitivity'));
    const ul = el('ul', 'sens-list');
    p.sensitivity.forEach((s) =>
      ul.appendChild(
        el('li', null, typeof s === 'string' ? s : JSON.stringify(s))
      )
    );
    c.appendChild(ul);
  }
  return c;
}

function bodyCostTax(p) {
  const c = el('div', 'insight-body');
  if (p.scenario_a) {
    c.appendChild(el('h4', 'sec-heading', 'Scenario A'));
    c.appendChild(renderObjectFlat(p.scenario_a));
  }
  if (p.scenario_b) {
    c.appendChild(el('h4', 'sec-heading', 'Scenario B'));
    c.appendChild(renderObjectFlat(p.scenario_b));
  }
  if (p.assumptions) {
    c.appendChild(el('h4', 'sec-heading', 'Assumptions'));
    c.appendChild(renderObjectFlat(p.assumptions));
  }
  if (p.estimated_impact) {
    c.appendChild(el('h4', 'sec-heading', 'Estimated impact'));
    const imp = el('div', 'impact-row');
    const units = p.estimated_impact.units || '';
    if (p.estimated_impact.range) {
      const r = p.estimated_impact.range;
      imp.appendChild(
        el('span', 'impact-val',
          r.map((v) => fmtVal(v, null, units)).join(' – '))
      );
    }
    if (units) {
      imp.appendChild(
        el('span', 'impact-unit',
          ' (' + units.replace(/_/g, ' ') + ')')
      );
    }
    c.appendChild(imp);
    if (DRAG_UNITS.has(units)) {
      c.appendChild(el('p', 'impact-note',
        'Proportion of value retained after expenses over the projection horizon.'));
    }
  }
  return c;
}

const BODY_RENDERERS = {
  diagnostic: bodyDiagnostic,
  benchmark_comparison: bodyBenchmark,
  scenario: bodyScenario,
  cost_tax: bodyCostTax,
};

const TEMPLATE_TITLES = {
  diagnostic: 'Diagnostic',
  benchmark_comparison: 'Benchmark comparison',
  scenario: 'Scenario analysis',
  cost_tax: 'Cost and tax impact',
};

/* ── Metric family identification (for impact linking) ── */

const METRIC_FAMILIES = {
  trailing_returns: 'trailing returns',
  rolling_excess_returns: 'rolling excess returns',
  drawdown: 'drawdown profile',
  cost_tax: 'cost and tax impact',
};

function identifyMetricFamily(compiled) {
  const t = compiled.template;
  if (t === 'cost_tax') return 'cost_tax';
  const sd = (compiled.payload && compiled.payload.supporting_data) || {};
  if ('trailing_returns' in sd) return 'trailing_returns';
  if ('excess_return_stats' in sd) return 'rolling_excess_returns';
  if ('fund_drawdown' in sd) return 'drawdown';
  return null; /* unknown — will not be highlighted */
}

/* evidence_strength lives on every card — mapped as a virtual family */
const EVIDENCE_FAMILY = 'evidence strength';

/* ── Main insight renderer ── */

function renderInsight(compiled) {
  const p = compiled.payload;
  const t = compiled.template;
  const card = el('article', 'insight');

  /* Tag card with metric family for impact linking */
  const family = identifyMetricFamily(compiled);
  if (family) {
    card.dataset.metric = family;
    card.dataset.metricLabel = METRIC_FAMILIES[family] || family;
  }

  /* Title — neutral, descriptive only */
  card.appendChild(el('h2', 'insight-title', TEMPLATE_TITLES[t] || t));

  /* Evidence — always first, always visible */
  card.appendChild(renderEvidenceRow(p));

  /* Template body */
  const body = (BODY_RENDERERS[t] || bodyDiagnostic)(p);
  card.appendChild(body);

  /* Method notes — never hidden, never collapsed */
  const lims = renderLimitations(p.limitations, 'Method notes');
  if (lims) card.appendChild(lims);

  /* Unavailable components */
  const unavail = renderUnavailable(p.unavailable_components);
  if (unavail) card.appendChild(unavail);

  return card;
}

function renderError(message) {
  insightsEl.innerHTML = '';
  const div = document.createElement('div');
  div.className = 'error';
  div.textContent = message;
  insightsEl.appendChild(div);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || data.error?.message || 'Request failed');
  }
  return data;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Running';

  try {
    const formData = new FormData(form);
    const body = JSON.parse(formData.get('portfolio_json'));
    body.user_country = formData.get('user_country');
    body.asset_market = formData.get('asset_market');
    body.serving_entity = formData.get('serving_entity');
    lastPortfolioValue = (body.holdings || []).reduce((sum, holding) => {
      return sum + Number(holding.market_value || 0);
    }, 0);

    const result = await postJson('/analytics/portfolio', body);
    gateStatus.textContent = result.gate.reason;
    result.insights.forEach((insight) => insightsEl.appendChild(renderInsight(insight)));
  } catch (error) {
    gateStatus.textContent = 'Blocked';
    renderError(error.message);
  }
});

scenarioButton.addEventListener('click', async () => {
  try {
    const result = await postJson('/scenarios/run', {
      subject_token: 'demo-user',
      portfolio_value: lastPortfolioValue,
      scenario_definition: { kind: 'standard', id: 'market_down_20' },
    });
    insightsEl.prepend(renderInsight(result.insight));
  } catch (error) {
    renderError(error.message);
  }
});

/* ── Ingestion report rendering ── */

function renderIngestionReport(report) {
  if (!report) return null;
  const section = el('article', 'insight ingestion-report');
  section.appendChild(el('h2', 'insight-title', 'Data ingestion report'));

  /* Source lineage */
  const lineage = el('div', 'ingestion-lineage');
  lineage.appendChild(renderKvRow('source', report.source || '—'));
  lineage.appendChild(renderKvRow('file', report.source_path || '—'));
  if (report.ingestion_timestamp) {
    lineage.appendChild(renderKvRow('timestamp', report.ingestion_timestamp));
  }
  if (report.license) {
    lineage.appendChild(renderKvRow('license', report.license));
  }
  if (report.mapping_label && report.mapping_label !== 'direct') {
    lineage.appendChild(renderKvRow('schema mapping', report.mapping_label));
  }
  section.appendChild(lineage);

  /* Series metadata */
  for (const [key, label] of [['fund_series', 'Fund series'], ['benchmark_series', 'Benchmark series']]) {
    const meta = report[key];
    if (!meta) continue;
    section.appendChild(el('h4', 'sec-heading', label));
    const stats = el('div', 'ingestion-stats');
    stats.appendChild(renderKvRow('input records', meta.input_records));
    stats.appendChild(renderKvRow('output points', meta.output_points));
    stats.appendChild(renderKvRow('rejected', meta.rejected_count));
    stats.appendChild(renderKvRow('duplicates merged', meta.duplicate_dates_merged));
    stats.appendChild(renderKvRow('anomalies flagged', meta.anomalies_flagged));
    stats.appendChild(renderKvRow('gaps detected', meta.gaps_detected));
    if (meta.date_range) {
      stats.appendChild(renderKvRow('date range',
        (meta.date_range.start || '?') + ' to ' + (meta.date_range.end || '?')));
    }
    section.appendChild(stats);
  }

  /* Ingestion limitations — labeled distinctly from analyzer limitations */
  const lims = report.ingestion_limitations;
  const limSection = renderLimitations(
    lims || [],
    'Data ingestion limitations',
    { allowEmpty: true },
  );
  if (limSection) section.appendChild(limSection);

  /* Impact links — connect data issues to affected metrics */
  const links = report.impact_links;
  if (links && links.length) {
    section.appendChild(el('h4', 'sec-heading', 'How data issues affect analysis'));
    const linkList = el('div', 'impact-links');
    links.forEach((link) => {
      const item = el('div', 'impact-link-item');
      item.dataset.targets = JSON.stringify(link.affected_metrics);
      item.setAttribute('role', 'button');
      item.setAttribute('tabindex', '0');

      const metrics = link.affected_metrics.join(', ');
      item.appendChild(el('span', 'impact-link-metrics', metrics));
      item.appendChild(el('span', 'impact-link-sep', ' — '));
      item.appendChild(el('span', 'impact-link-text', link.explanation));

      item.addEventListener('click', () => highlightAffected(link.affected_metrics, link.issue, item));
      item.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          highlightAffected(link.affected_metrics, link.issue, item);
        }
      });

      item.addEventListener('mouseenter', () => previewAffected(link.affected_metrics));
      item.addEventListener('mouseleave', clearPreview);

      linkList.appendChild(item);
    });
    section.appendChild(linkList);
  }

  return section;
}

/* ── Impact → insight highlighting ── */

/* Human-readable issue labels for context header */
const ISSUE_LABELS = {
  short_history: 'Short history',
  records_rejected: 'Rejected records',
  duplicates_merged: 'Duplicates merged',
  gaps_detected: 'Gaps',
  extreme_values: 'Extreme values',
};

function clearHighlights() {
  document.querySelectorAll('.insight-affected').forEach(
    (node) => node.classList.remove('insight-affected'));
  document.querySelectorAll('.impact-link-active').forEach(
    (node) => node.classList.remove('impact-link-active'));
  document.querySelectorAll('.impact-link-was-active').forEach(
    (node) => node.classList.remove('impact-link-was-active'));
  const existing = document.querySelector('.impact-context-header');
  if (existing) existing.remove();
}

function findMatchingCards(metricFamilies) {
  const matchesEvidence = metricFamilies.includes(EVIDENCE_FAMILY);
  const cards = insightsEl.querySelectorAll('.insight[data-metric]');
  const matches = [];
  cards.forEach((card) => {
    const cardFamily = METRIC_FAMILIES[card.dataset.metric] || card.dataset.metric;
    if (matchesEvidence || metricFamilies.includes(cardFamily)) {
      matches.push(card);
    }
  });
  return matches;
}

function highlightAffected(metricFamilies, issueKey, clickedItem) {
  const wasActive = clickedItem.classList.contains('impact-link-was-active');
  clearHighlights();

  /* Toggle: clicking same link again just clears */
  if (wasActive) return;

  clickedItem.classList.add('impact-link-active', 'impact-link-was-active');

  /* Context header — anchored above insight cards */
  const label = ISSUE_LABELS[issueKey] || issueKey;
  const header = el('div', 'impact-context-header',
    'Showing insights related to: ' + label);
  const firstInsight = insightsEl.querySelector('.insight');
  if (firstInsight) {
    insightsEl.insertBefore(header, firstInsight);
  }

  /* Find and highlight matching insight cards */
  const matches = findMatchingCards(metricFamilies);
  matches.forEach((card) => card.classList.add('insight-affected'));

  /* Scroll to first affected card (smooth, only when off-screen) */
  if (matches.length) {
    matches[0].scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

/* ── Hover preview — light outline, no scroll ── */

function previewAffected(metricFamilies) {
  /* Don't preview if a link is already active (click takes priority) */
  if (document.querySelector('.impact-link-active')) return;
  const matches = findMatchingCards(metricFamilies);
  matches.forEach((card) => card.classList.add('insight-hover-preview'));
}

function clearPreview() {
  document.querySelectorAll('.insight-hover-preview').forEach(
    (node) => node.classList.remove('insight-hover-preview'));
}

/* ── From-source demo ── */

function runFromSource(symbol, fundLabel) {
  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Running from source';
  postJson('/analytics/mutual-fund/from-source', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    source: 'csv_sample',
    symbol: symbol,
    fund_name: fundLabel,
    benchmark_name: fundLabel + ' Benchmark',
    category: 'Equity',
    expense_ratio_pct: 1.5,
    rolling_window_points: 4,
    rolling_step_points: 1,
    rolling_min_windows: 1,
  }).then((result) => {
    gateStatus.textContent = result.gate.reason;

    /* Ingestion report first — data truth before analysis */
    const reportEl = renderIngestionReport(result.ingestion_report);
    if (reportEl) insightsEl.appendChild(reportEl);

    /* Then analyzer insights */
    (result.insights || []).forEach((insight) =>
      insightsEl.appendChild(renderInsight(insight))
    );
  }).catch((error) => {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  });
}

fromSourceClean.addEventListener('click', () =>
  runFromSource('clean_mf', 'Clean Fund')
);

fromSourceMessy.addEventListener('click', () =>
  runFromSource('messy_mf', 'Messy Fund')
);


/* ═══════════════════════════════════════════════════════════════
   Fund Discovery — search → select → fetch → analyze
   ═══════════════════════════════════════════════════════════════ */

const discoverSearchInput = document.getElementById('discoverSearch');
const discoverSearchBtn = document.getElementById('discoverSearchBtn');
const registryRefreshBtn = document.getElementById('registryRefreshBtn');
const registryStatusEl = document.getElementById('registryStatus');
const discoverResultsEl = document.getElementById('discoverResults');

/* ── Registry refresh ── */

registryRefreshBtn.addEventListener('click', () => {
  registryStatusEl.textContent = 'Downloading AMFI scheme list…';
  registryRefreshBtn.disabled = true;
  postJson('/discover/refresh-registry', {})
    .then((data) => {
      registryStatusEl.textContent = data.schemes_loaded + ' schemes loaded';
      registryRefreshBtn.disabled = false;
    })
    .catch((err) => {
      registryStatusEl.textContent = 'Refresh failed: ' + err.message;
      registryRefreshBtn.disabled = false;
    });
});

/* ── Check registry status on load ── */

fetch('/discover/registry-status')
  .then((r) => r.json())
  .then((data) => {
    if (data.loaded) {
      registryStatusEl.textContent = data.count + ' schemes';
    } else {
      registryStatusEl.textContent = 'Registry empty — click Refresh';
    }
  })
  .catch(() => {
    registryStatusEl.textContent = '';
  });

/* ── Search ── */

let searchDebounce = null;

function runDiscoverSearch() {
  const q = discoverSearchInput.value.trim();
  if (q.length < 2) {
    discoverResultsEl.innerHTML = '';
    return;
  }
  fetch('/discover/search?q=' + encodeURIComponent(q) + '&max_results=15')
    .then((r) => r.json())
    .then((data) => {
      renderDiscoverResults(data.results || []);
    })
    .catch((err) => {
      discoverResultsEl.innerHTML = '';
      discoverResultsEl.appendChild(
        el('div', 'discover-error', 'Search error: ' + err.message)
      );
    });
}

discoverSearchBtn.addEventListener('click', runDiscoverSearch);

discoverSearchInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    runDiscoverSearch();
  }
});

discoverSearchInput.addEventListener('input', () => {
  clearTimeout(searchDebounce);
  searchDebounce = setTimeout(runDiscoverSearch, 400);
});

/* ── Render search results ── */

/* Compare selection state */
const compareSelection = new Map(); /* scheme_code → scheme object */
const compareBar = document.getElementById('compareBar');
const compareCountEl = document.getElementById('compareCount');
const compareBtn = document.getElementById('compareBtn');
const compareClearBtn = document.getElementById('compareClearBtn');
const sipToggleBtn = document.getElementById('sipToggleBtn');
const sipControls = document.getElementById('sipControls');
const sipWeightsEl = document.getElementById('sipWeights');
const sipRunBtn = document.getElementById('sipRunBtn');
const sipAmountInput = document.getElementById('sipAmount');
const sipWindowInput = document.getElementById('sipWindowMonths');
const portfolioToggleBtn = document.getElementById('portfolioToggleBtn');
const portfolioControls = document.getElementById('portfolioControls');
const portfolioWeightsEl = document.getElementById('portfolioWeights');
const portfolioRunBtn = document.getElementById('portfolioRunBtn');
const portfolioWindowInput = document.getElementById('portfolioWindow');
const portfolioStepInput = document.getElementById('portfolioStep');
const evaluateToggleBtn = document.getElementById('evaluateToggleBtn');
const evaluateControls = document.getElementById('evaluateControls');
const evaluateWeightsEl = document.getElementById('evaluateWeights');
const evaluateRunBtn = document.getElementById('evaluateRunBtn');

function updateCompareBar() {
  const n = compareSelection.size;
  compareBar.style.display = n > 0 ? 'flex' : 'none';
  compareCountEl.textContent = n + ' selected';
  compareBtn.disabled = n < 2;
  compareBtn.textContent = n >= 2
    ? 'Compare Selected (' + n + ')'
    : 'Compare Selected';
  sipToggleBtn.disabled = n < 1;
  portfolioToggleBtn.disabled = n < 2;
  evaluateToggleBtn.disabled = n < 2;
  /* Update checkbox states */
  document.querySelectorAll('.discover-check').forEach((cb) => {
    const code = Number(cb.dataset.code);
    cb.checked = compareSelection.has(code);
    cb.disabled = !compareSelection.has(code) && n >= 5;
  });
  /* Update SIP weight inputs if panel is visible */
  if (sipControls.style.display !== 'none') {
    renderSipWeights();
  }
  /* Update portfolio weight inputs if panel is visible */
  if (portfolioControls.style.display !== 'none') {
    renderPortfolioWeights();
  }
  /* Update evaluate weight inputs if panel is visible */
  if (evaluateControls.style.display !== 'none') {
    renderEvaluateWeights();
  }
}

compareClearBtn.addEventListener('click', () => {
  compareSelection.clear();
  sipControls.style.display = 'none';
  evaluateControls.style.display = 'none';
  portfolioControls.style.display = 'none';
  updateCompareBar();
});

/* ── SIP toggle + weight inputs ── */

sipToggleBtn.addEventListener('click', () => {
  const visible = sipControls.style.display !== 'none';
  sipControls.style.display = visible ? 'none' : 'block';
  if (!visible) renderSipWeights();
});

function renderSipWeights() {
  sipWeightsEl.innerHTML = '';
  const n = compareSelection.size;
  if (n === 0) return;
  const equalWeight = Math.round(100 / n);
  compareSelection.forEach((scheme, code) => {
    const row = el('div', 'sip-weight-row');
    const label = el('span', 'sip-weight-label',
      (scheme.scheme_name || 'Scheme ' + code).substring(0, 40));
    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'sip-weight-input';
    input.min = '0';
    input.max = '100';
    input.value = String(equalWeight);
    input.dataset.code = code;
    const pct = el('span', 'sip-weight-pct', '%');
    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(pct);
    sipWeightsEl.appendChild(row);
  });
}

function renderDiscoverResults(results) {
  discoverResultsEl.innerHTML = '';
  if (!results.length) {
    discoverResultsEl.appendChild(
      el('div', 'discover-empty', 'No matching schemes found.')
    );
    return;
  }
  const table = el('table', 'discover-table');
  const thead = el('thead');
  const hrow = el('tr');
  ['', 'Code', 'Scheme Name', 'Fund House', 'Category', 'NAV', ''].forEach((h) =>
    hrow.appendChild(el('th', '', h))
  );
  thead.appendChild(hrow);
  table.appendChild(thead);

  const tbody = el('tbody');
  results.forEach((scheme) => {
    const row = el('tr', 'discover-row');
    /* Checkbox for compare selection */
    const checkTd = el('td', 'discover-check-cell');
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.className = 'discover-check';
    cb.dataset.code = scheme.scheme_code;
    cb.checked = compareSelection.has(scheme.scheme_code);
    cb.disabled = !compareSelection.has(scheme.scheme_code) && compareSelection.size >= 5;
    cb.addEventListener('change', () => {
      if (cb.checked) {
        compareSelection.set(scheme.scheme_code, scheme);
      } else {
        compareSelection.delete(scheme.scheme_code);
      }
      updateCompareBar();
    });
    checkTd.appendChild(cb);
    row.appendChild(checkTd);
    row.appendChild(el('td', 'discover-code', String(scheme.scheme_code)));
    row.appendChild(el('td', 'discover-name', scheme.scheme_name));
    row.appendChild(el('td', 'discover-fh', scheme.fund_house));
    row.appendChild(el('td', 'discover-cat', scheme.scheme_category));
    row.appendChild(
      el('td', 'discover-nav',
        scheme.latest_nav != null ? String(scheme.latest_nav) : '—')
    );
    const actionTd = el('td', 'discover-action');
    const analyzeBtn = el('button', 'btn-analyze', 'Analyze');
    analyzeBtn.addEventListener('click', () =>
      runDiscoverAnalyze(scheme)
    );
    actionTd.appendChild(analyzeBtn);
    row.appendChild(actionTd);
    tbody.appendChild(row);
  });
  table.appendChild(tbody);
  discoverResultsEl.appendChild(table);
}

/* ── Fetch & Analyze a selected scheme ── */

function runDiscoverAnalyze(scheme) {
  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Fetching NAV for ' + scheme.scheme_name + '…';

  postJson('/discover/fetch-and-analyze', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    scheme_code: scheme.scheme_code,
    fund_name: scheme.scheme_name,
    category: scheme.scheme_category || 'Unknown',
    expense_ratio_pct: 0,
    rolling_window_points: 4,
    rolling_step_points: 1,
    rolling_min_windows: 1,
  }).then((result) => {
    gateStatus.textContent = result.gate.reason;

    /* Ingestion report first */
    const reportEl = renderIngestionReport(result.ingestion_report);
    if (reportEl) insightsEl.appendChild(reportEl);

    /* Analyzer insights */
    (result.insights || []).forEach((insight) =>
      insightsEl.appendChild(renderInsight(insight))
    );
  }).catch((error) => {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  });
}


/* ═══════════════════════════════════════════════════════════════
   Fund Comparison — multi-select → compare side-by-side
   ═══════════════════════════════════════════════════════════════ */

compareBtn.addEventListener('click', runCompare);

function runCompare() {
  if (compareSelection.size < 2) return;
  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Comparing ' + compareSelection.size + ' funds…';

  const funds = [];
  compareSelection.forEach((scheme) => {
    funds.push({
      scheme_code: scheme.scheme_code,
      fund_name: scheme.scheme_name || '',
      category: scheme.scheme_category || 'Unknown',
      expense_ratio_pct: 0,
    });
  });

  postJson('/analytics/compare', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    funds: funds,
    rolling_window_points: 60,
    rolling_step_points: 5,
  }).then((result) => {
    gateStatus.textContent = result.gate.reason;
    renderCompareResult(result);
  }).catch((error) => {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  });
}

function renderCompareResult(result) {
  insightsEl.innerHTML = '';

  /* Alignment quality header */
  const aq = result.alignment_quality;
  if (aq) {
    const header = el('article', 'insight compare-header');
    header.appendChild(el('h2', 'insight-title', 'Fund Comparison'));
    header.appendChild(el('p', 'obs',
      'Computed on ' + aq.aligned_points + ' common observation dates across '
      + (result.funds || []).length + ' funds.'));
    const qualRow = el('div', 'evidence-row');
    qualRow.appendChild(el('span', 'evidence-label', 'Alignment evidence: '));
    qualRow.appendChild(evidenceBadge(aq.evidence));
    if (aq.aligned_pct != null) {
      qualRow.appendChild(el('span', 'evidence-sep', ' · '));
      qualRow.appendChild(el('span', 'evidence-label',
        aq.aligned_pct.toFixed(1) + '% of the smallest series'));
    }
    header.appendChild(qualRow);
    insightsEl.appendChild(header);
  }

  /* Comparison insights — cross-fund diagnostics */
  const compInsights = result.comparison_insights || [];
  if (compInsights.length) {
    const compSection = el('div', 'compare-insights-section');
    compSection.appendChild(el('h3', 'compare-section-heading', 'Cross-Fund Analysis'));
    compInsights.forEach((insight) =>
      compSection.appendChild(renderInsight(insight))
    );
    insightsEl.appendChild(compSection);
  }

  /* Per-fund details */
  const funds = result.funds || [];
  if (funds.length) {
    const fundSection = el('div', 'compare-per-fund-section');
    fundSection.appendChild(el('h3', 'compare-section-heading', 'Per-Fund Details'));

    funds.forEach((fund) => {
      const fundCard = el('div', 'compare-fund-card');
      fundCard.appendChild(el('h4', 'compare-fund-name',
        fund.name + ' (' + fund.scheme_code + ')'));
      if (fund.category && fund.category !== 'Unknown') {
        fundCard.appendChild(el('span', 'compare-fund-category', fund.category));
      }
      /* Data quality */
      if (fund.data_quality) {
        const dqRow = el('div', 'evidence-row');
        dqRow.appendChild(el('span', 'evidence-label', 'Data quality: '));
        dqRow.appendChild(evidenceBadge(fund.data_quality.evidence_strength));
        fundCard.appendChild(dqRow);
      }
      /* Insights for this fund */
      (fund.insights || []).forEach((insight) =>
        fundCard.appendChild(renderInsight(insight))
      );
      fundSection.appendChild(fundCard);
    });

    insightsEl.appendChild(fundSection);
  }

  /* Suppressed comparison insights */
  const suppressed = result.suppressed_comparison_insights || [];
  if (suppressed.length) {
    const suppSection = el('div', 'compare-suppressed');
    suppSection.appendChild(el('h4', 'lim-heading',
      suppressed.length + ' comparison insight(s) suppressed'));
    const list = el('ul', 'lim-list');
    suppressed.forEach((s) => {
      list.appendChild(el('li', null, (s.type || '?') + ': ' + s.reason));
    });
    suppSection.appendChild(list);
    insightsEl.appendChild(suppSection);
  }
}


/* ═══════════════════════════════════════════════════════════════
   SIP Simulator — rolling historical SIP outcomes
   ═══════════════════════════════════════════════════════════════ */

sipRunBtn.addEventListener('click', runSipSimulation);

function runSipSimulation() {
  if (compareSelection.size < 1) return;

  /* Collect weights from inputs */
  const weightInputs = sipWeightsEl.querySelectorAll('.sip-weight-input');
  let totalWeight = 0;
  const funds = [];
  weightInputs.forEach((input) => {
    const code = Number(input.dataset.code);
    const pct = Number(input.value) || 0;
    totalWeight += pct;
    const scheme = compareSelection.get(code);
    funds.push({
      scheme_code: code,
      fund_name: scheme ? scheme.scheme_name || '' : '',
      category: scheme ? scheme.scheme_category || 'Unknown' : 'Unknown',
      expense_ratio_pct: 0,
      weight: pct / 100,
    });
  });

  if (Math.abs(totalWeight - 100) > 1) {
    renderError('Fund weights must sum to 100%. Currently: ' + totalWeight + '%');
    return;
  }

  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Simulating SIP across ' + funds.length + ' fund(s)…';

  postJson('/analytics/sip', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    funds: funds,
    monthly_amount: Number(sipAmountInput.value) || 10000,
    rolling_window_months: Number(sipWindowInput.value) || 36,
    step_months: 1,
  }).then((result) => {
    gateStatus.textContent = result.gate.reason;
    renderSipResult(result);
  }).catch((error) => {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  });
}

function renderSipResult(result) {
  insightsEl.innerHTML = '';

  /* Simulation meta header */
  const meta = result.simulation_meta;
  const aq = result.alignment_quality;
  if (meta) {
    const header = el('article', 'insight sip-header');
    header.appendChild(el('h2', 'insight-title', 'SIP Simulation'));
    header.appendChild(el('p', 'obs',
      meta.window_count + ' rolling ' + meta.window_months + '-month windows '
      + 'computed on ' + (aq ? aq.aligned_points : '?') + ' common dates.'));
    if (aq) {
      const qualRow = el('div', 'evidence-row');
      qualRow.appendChild(el('span', 'evidence-label', 'Data evidence: '));
      qualRow.appendChild(evidenceBadge(aq.evidence));
      header.appendChild(qualRow);
    }
    insightsEl.appendChild(header);
  }

  /* SIP insights */
  const sipInsights = result.sip_insights || [];
  if (sipInsights.length) {
    const section = el('div', 'sip-insights-section');
    section.appendChild(el('h3', 'compare-section-heading', 'SIP Analysis'));
    sipInsights.forEach((insight) =>
      section.appendChild(renderInsight(insight))
    );
    insightsEl.appendChild(section);
  }

  /* Per-fund details */
  const funds = result.funds || [];
  if (funds.length) {
    const fundSection = el('div', 'compare-per-fund-section');
    fundSection.appendChild(el('h3', 'compare-section-heading', 'Per-Fund Details'));
    funds.forEach((fund) => {
      const fundCard = el('div', 'compare-fund-card');
      fundCard.appendChild(el('h4', 'compare-fund-name',
        fund.name + ' (' + fund.scheme_code + ')'));
      if (fund.data_quality) {
        const dqRow = el('div', 'evidence-row');
        dqRow.appendChild(el('span', 'evidence-label', 'Data quality: '));
        dqRow.appendChild(evidenceBadge(fund.data_quality.evidence_strength));
        fundCard.appendChild(dqRow);
      }
      (fund.insights || []).forEach((insight) =>
        fundCard.appendChild(renderInsight(insight))
      );
      fundSection.appendChild(fundCard);
    });
    insightsEl.appendChild(fundSection);
  }

  /* Suppressed */
  const suppressed = result.suppressed_sip_insights || [];
  if (suppressed.length) {
    const suppSection = el('div', 'compare-suppressed');
    suppSection.appendChild(el('h4', 'lim-heading',
      suppressed.length + ' SIP insight(s) suppressed'));
    const list = el('ul', 'lim-list');
    suppressed.forEach((s) => {
      list.appendChild(el('li', null, (s.type || '?') + ': ' + s.reason));
    });
    suppSection.appendChild(list);
    insightsEl.appendChild(suppSection);
  }
}

/* ── Portfolio Aggregator ── */

portfolioToggleBtn.addEventListener('click', () => {
  const visible = portfolioControls.style.display !== 'none';
  portfolioControls.style.display = visible ? 'none' : 'block';
  if (!visible) renderPortfolioWeights();
});

function renderPortfolioWeights() {
  portfolioWeightsEl.innerHTML = '';
  const n = compareSelection.size;
  if (n === 0) return;
  const equalWeight = Math.round(100 / n);
  compareSelection.forEach((scheme, code) => {
    const row = el('div', 'portfolio-weight-row');
    const label = el('span', 'portfolio-weight-label',
      (scheme.scheme_name || 'Scheme ' + code).substring(0, 40));
    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'portfolio-weight-input';
    input.min = '0';
    input.max = '100';
    input.value = String(equalWeight);
    input.dataset.code = code;
    const pct = el('span', 'portfolio-weight-pct', '%');
    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(pct);
    portfolioWeightsEl.appendChild(row);
  });
}

portfolioRunBtn.addEventListener('click', runPortfolioAggregate);

function runPortfolioAggregate() {
  if (compareSelection.size < 2) return;

  const weightInputs = portfolioWeightsEl.querySelectorAll('.portfolio-weight-input');
  let totalWeight = 0;
  const funds = [];
  weightInputs.forEach((input) => {
    const code = Number(input.dataset.code);
    const pct = Number(input.value) || 0;
    totalWeight += pct;
    const scheme = compareSelection.get(code);
    funds.push({
      scheme_code: code,
      fund_name: scheme ? scheme.scheme_name || '' : '',
      category: scheme ? scheme.scheme_category || 'Unknown' : 'Unknown',
      expense_ratio_pct: 0,
      weight: pct / 100,
    });
  });

  if (Math.abs(totalWeight - 100) > 1) {
    renderError('Fund weights must sum to 100%. Currently: ' + totalWeight + '%');
    return;
  }

  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Aggregating portfolio with ' + funds.length + ' fund(s)…';

  postJson('/analytics/portfolio-aggregate', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    funds: funds,
    rolling_window_points: Number(portfolioWindowInput.value) || 252,
    rolling_step_points: Number(portfolioStepInput.value) || 21,
  }).then((result) => {
    gateStatus.textContent = result.gate.reason;
    renderPortfolioResult(result);
  }).catch((error) => {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  });
}

function renderPortfolioResult(result) {
  insightsEl.innerHTML = '';

  /* Header with alignment quality */
  const aq = result.alignment_quality;
  const header = el('article', 'insight portfolio-header');
  header.appendChild(el('h2', 'insight-title', 'Portfolio Aggregation'));
  if (aq) {
    header.appendChild(el('p', 'obs',
      'Computed on ' + aq.aligned_points + ' common dates across '
      + (result.funds ? result.funds.length : '?') + ' fund(s).'));
    const qualRow = el('div', 'evidence-row');
    qualRow.appendChild(el('span', 'evidence-label', 'Data evidence: '));
    qualRow.appendChild(evidenceBadge(aq.evidence));
    header.appendChild(qualRow);
  }
  insightsEl.appendChild(header);

  /* Portfolio insights */
  const portfolioInsights = result.portfolio_insights || [];
  if (portfolioInsights.length) {
    const section = el('div', 'portfolio-insights-section');
    section.appendChild(el('h3', 'compare-section-heading', 'Portfolio Analysis'));
    portfolioInsights.forEach((insight) =>
      section.appendChild(renderInsight(insight))
    );
    insightsEl.appendChild(section);
  }

  /* Per-fund details */
  const funds = result.funds || [];
  if (funds.length) {
    const fundSection = el('div', 'compare-per-fund-section');
    fundSection.appendChild(el('h3', 'compare-section-heading', 'Per-Fund Details'));
    funds.forEach((fund) => {
      const fundCard = el('div', 'compare-fund-card');
      fundCard.appendChild(el('h4', 'compare-fund-name',
        fund.name + ' (' + fund.scheme_code + ')'));
      if (fund.data_quality) {
        const dqRow = el('div', 'evidence-row');
        dqRow.appendChild(el('span', 'evidence-label', 'Data quality: '));
        dqRow.appendChild(evidenceBadge(fund.data_quality.evidence_strength));
        fundCard.appendChild(dqRow);
      }
      (fund.insights || []).forEach((insight) =>
        fundCard.appendChild(renderInsight(insight))
      );
      fundSection.appendChild(fundCard);
    });
    insightsEl.appendChild(fundSection);
  }

  /* Suppressed */
  const suppressed = result.suppressed_portfolio_insights || [];
  if (suppressed.length) {
    const suppSection = el('div', 'compare-suppressed');
    suppSection.appendChild(el('h4', 'lim-heading',
      suppressed.length + ' portfolio insight(s) suppressed'));
    const list = el('ul', 'lim-list');
    suppressed.forEach((s) => {
      list.appendChild(el('li', null, (s.type || '?') + ': ' + s.reason));
    });
    suppSection.appendChild(list);
    insightsEl.appendChild(suppSection);
  }
}

/* ── Portfolio Evaluation ── */

evaluateToggleBtn.addEventListener('click', () => {
  const visible = evaluateControls.style.display !== 'none';
  evaluateControls.style.display = visible ? 'none' : 'block';
  if (!visible) renderEvaluateWeights();
});

function renderEvaluateWeights() {
  evaluateWeightsEl.innerHTML = '';
  const n = compareSelection.size;
  if (n === 0) return;
  const equalWeight = Math.round(100 / n);
  compareSelection.forEach((scheme, code) => {
    const row = el('div', 'evaluate-weight-row');
    const label = el('span', 'evaluate-weight-label',
      (scheme.scheme_name || 'Scheme ' + code).substring(0, 40));
    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'evaluate-weight-input';
    input.min = '0';
    input.max = '100';
    input.value = String(equalWeight);
    input.dataset.code = code;
    const pct = el('span', 'evaluate-weight-pct', '%');
    row.appendChild(label);
    row.appendChild(input);
    row.appendChild(pct);
    evaluateWeightsEl.appendChild(row);
  });
}

evaluateRunBtn.addEventListener('click', runPortfolioEvaluate);

function runPortfolioEvaluate() {
  if (compareSelection.size < 2) return;

  const weightInputs = evaluateWeightsEl.querySelectorAll('.evaluate-weight-input');
  let totalWeight = 0;
  const funds = [];
  weightInputs.forEach((input) => {
    const code = Number(input.dataset.code);
    const pct = Number(input.value) || 0;
    totalWeight += pct;
    const scheme = compareSelection.get(code);
    funds.push({
      scheme_code: code,
      fund_name: scheme ? scheme.scheme_name || '' : '',
      category: scheme ? scheme.scheme_category || 'Unknown' : 'Unknown',
      expense_ratio_pct: 0,
      weight: pct / 100,
    });
  });

  if (Math.abs(totalWeight - 100) > 1) {
    renderError('Fund weights must sum to 100%. Currently: ' + totalWeight + '%');
    return;
  }

  /* Collect constraint inputs */
  const constraints = {};
  const dd = document.getElementById('evalMaxDrawdown').value;
  if (dd !== '') constraints.max_drawdown_pct = Number(dd);
  const rec = document.getElementById('evalMaxRecovery').value;
  if (rec !== '') constraints.max_recovery_days = Number(rec);
  const cagr = document.getElementById('evalMinCagr').value;
  if (cagr !== '') constraints.min_median_rolling_cagr_pct = Number(cagr);
  const vol = document.getElementById('evalMaxVol').value;
  if (vol !== '') constraints.max_volatility_pct = Number(vol);
  const corr = document.getElementById('evalMaxCorr').value;
  if (corr !== '') constraints.max_correlation = Number(corr);
  const hhi = document.getElementById('evalMaxHhi').value;
  if (hhi !== '') constraints.max_concentration_hhi = Number(hhi);
  const fundDd = document.getElementById('evalMaxFundDd').value;
  if (fundDd !== '') constraints.max_single_fund_drawdown_pct = Number(fundDd);

  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Evaluating portfolio\u2026';

  postJson('/analytics/portfolio-evaluate', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    funds: funds,
    rolling_window_points: Number(portfolioWindowInput.value) || 252,
    rolling_step_points: Number(portfolioStepInput.value) || 21,
    constraints: constraints,
  }).then((result) => {
    gateStatus.textContent = result.gate.reason;
    renderEvaluationResult(result);
  }).catch((error) => {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  });
}

function renderEvaluationResult(result) {
  insightsEl.innerHTML = '';

  const ev = result.evaluation;
  if (!ev) {
    insightsEl.appendChild(el('p', 'obs', 'No evaluation data returned.'));
    return;
  }

  /* Summary card */
  const summary = ev.summary || {};
  const verdictClass = summary.verdict === 'ALL_PASS' ? 'eval-pass'
    : summary.verdict === 'FAIL' ? 'eval-fail' : 'eval-incomplete';
  const header = el('article', 'insight eval-summary ' + verdictClass);
  header.appendChild(el('h2', 'insight-title', 'Evaluation Verdict'));
  header.appendChild(el('span', 'eval-verdict-badge ' + verdictClass, summary.verdict || '?'));
  const stats = el('div', 'eval-stats');
  stats.appendChild(el('span', 'eval-stat', 'Checks: ' + (summary.total_checks || 0)));
  stats.appendChild(el('span', 'eval-stat eval-stat-pass', 'Passed: ' + (summary.passed || 0)));
  stats.appendChild(el('span', 'eval-stat eval-stat-fail', 'Failed: ' + (summary.failed || 0)));
  if (summary.insufficient_data > 0) {
    stats.appendChild(el('span', 'eval-stat eval-stat-nodata',
      'No data: ' + summary.insufficient_data));
  }
  if (summary.flag_count > 0) {
    stats.appendChild(el('span', 'eval-stat eval-stat-flag',
      'Flags: ' + summary.flag_count));
  }
  header.appendChild(stats);
  insightsEl.appendChild(header);

  /* Constraint checks */
  const checks = ev.checks || [];
  if (checks.length) {
    const section = el('div', 'eval-checks-section');
    section.appendChild(el('h3', 'compare-section-heading', 'Constraint Checks'));
    const table = el('table', 'eval-checks-table');
    const thead = el('thead');
    const headRow = el('tr');
    ['Constraint', 'Status', 'Observed', 'Threshold'].forEach((h) =>
      headRow.appendChild(el('th', null, h))
    );
    thead.appendChild(headRow);
    table.appendChild(thead);
    const tbody = el('tbody');
    checks.forEach((c) => {
      const row = el('tr', 'eval-check-row eval-check-' + (c.status || '').toLowerCase());
      row.appendChild(el('td', 'eval-check-name', c.name || '?'));
      const statusCell = el('td');
      const badge = el('span', 'eval-status-badge eval-status-' + (c.status || '').toLowerCase(),
        c.status || '?');
      statusCell.appendChild(badge);
      row.appendChild(statusCell);
      row.appendChild(el('td', 'eval-check-value',
        c.observed !== null && c.observed !== undefined ? String(c.observed) : '\u2014'));
      row.appendChild(el('td', 'eval-check-value', String(c.threshold)));
      tbody.appendChild(row);
      /* Show why_failed row if present */
      if (c.why && c.status !== 'PASS') {
        const whyRow = el('tr', 'eval-why-row');
        const whyCell = el('td', 'eval-why-cell', c.why);
        whyCell.setAttribute('colspan', '4');
        whyRow.appendChild(whyCell);
        tbody.appendChild(whyRow);
      }
    });
    table.appendChild(tbody);
    section.appendChild(table);
    insightsEl.appendChild(section);
  }

  /* Red flags */
  const flags = ev.flags || [];
  if (flags.length) {
    const flagSection = el('div', 'eval-flags-section');
    flagSection.appendChild(el('h3', 'compare-section-heading', 'Structural Flags'));
    flags.forEach((f) => {
      const card = el('div', 'eval-flag-card');
      card.appendChild(el('span', 'eval-flag-icon', '\u26a0'));
      card.appendChild(el('span', 'eval-flag-name', f.flag || ''));
      card.appendChild(el('span', 'eval-flag-detail', f.detail || ''));
      flagSection.appendChild(card);
    });
    insightsEl.appendChild(flagSection);
  }
}
