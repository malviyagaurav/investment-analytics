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

/* ── Demo panel toggle ── */

const demoToggle = document.getElementById('demoToggle');
const demoPanel = document.getElementById('demoPanel');

demoToggle.addEventListener('click', () => {
  const hidden = demoPanel.style.display === 'none';
  demoPanel.style.display = hidden ? 'flex' : 'none';
  demoToggle.textContent = hidden ? 'Test / Demo ▾' : 'Test / Demo ▸';
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

/* ── Summary block rendering ── */

function renderSummaryBlock(summary) {
  if (!summary) return null;
  const m = summary.metrics || {};
  const interp = summary.interpretation || [];
  if (!Object.keys(m).length && !interp.length) return null;

  const card = el('article', 'insight summary-block');

  card.appendChild(el('h2', 'insight-title', 'Key metrics'));

  /* Metric chips */
  const chips = el('div', 'summary-chips');

  if (m.trailing_return_period) {
    chips.appendChild(summaryChip(
      m.trailing_return_period + ' Fund CAGR',
      fmtPct(m.fund_cagr_pct),
    ));
    if (m.benchmark_type !== 'self') {
      chips.appendChild(summaryChip(
        m.trailing_return_period + ' Benchmark CAGR',
        fmtPct(m.benchmark_cagr_pct),
      ));
      chips.appendChild(summaryChip(
        'Difference',
        fmtDiff(m.cagr_difference_pct),
        m.cagr_difference_pct >= 0 ? 'positive' : 'negative',
      ));
    }
  }
  if (m.rolling_hit_ratio_pct != null && m.benchmark_type !== 'self') {
    chips.appendChild(summaryChip(
      'Rolling hit ratio',
      fmtPct(m.rolling_hit_ratio_pct),
      m.rolling_hit_ratio_pct >= 50 ? 'positive' : 'negative',
    ));
  }
  if (m.max_drawdown_pct != null) {
    chips.appendChild(summaryChip(
      'Max drawdown',
      fmtPct(m.max_drawdown_pct),
      'neutral',
    ));
  }
  if (chips.children.length) card.appendChild(chips);

  /* Interpretation bullets */
  if (interp.length) {
    const list = el('ul', 'summary-interpretation');
    interp.forEach((text) => list.appendChild(el('li', '', text)));
    card.appendChild(list);
  }

  return card;
}

function summaryChip(label, value, tone) {
  const chip = el('div', 'summary-chip' + (tone ? ' chip-' + tone : ''));
  chip.appendChild(el('span', 'chip-label', label));
  chip.appendChild(el('span', 'chip-value', value));
  return chip;
}

function fmtPct(v) {
  if (v == null) return '—';
  return Number(v).toFixed(2) + '%';
}

function fmtDiff(v) {
  if (v == null) return '—';
  const n = Number(v);
  return (n >= 0 ? '+' : '') + n.toFixed(2) + ' pp';
}

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

    /* Summary block — key metrics + interpretation at top */
    const summaryEl = renderSummaryBlock(result.summary);
    if (summaryEl) insightsEl.appendChild(summaryEl);

    /* Ingestion report */
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


/* ═══════════════════════════════════════════════════════════════
   Category Ranking — pairwise dominance across all category funds
   ═══════════════════════════════════════════════════════════════ */

const rankCategorySelect = document.getElementById('rankCategory');
const rankRunBtn = document.getElementById('rankRunBtn');
const rankResultsEl = document.getElementById('rankResults');

rankCategorySelect.addEventListener('change', () => {
  rankRunBtn.disabled = !rankCategorySelect.value;
});

rankRunBtn.addEventListener('click', runCategoryRank);

function runCategoryRank() {
  const category = rankCategorySelect.value;
  if (!category) return;

  rankResultsEl.innerHTML = '';
  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Ranking ' + rankCategorySelect.selectedOptions[0].text + ' funds…';
  rankRunBtn.disabled = true;

  postJson('/analytics/rank-category', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    category: category,
  }).then((result) => {
    gateStatus.textContent = result.gate.reason;
    renderRankingResult(result.ranking);
  }).catch((error) => {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  }).finally(() => {
    rankRunBtn.disabled = false;
  });
}

function renderRankingResult(ranking) {
  if (!ranking) return;
  insightsEl.innerHTML = '';

  /* Header card */
  const header = el('article', 'insight ranking-header');
  header.appendChild(el('h2', 'insight-title', 'Category Ranking: ' + ranking.category.replace('Equity Scheme - ', '')));

  const meta = el('div', 'ranking-meta');
  meta.appendChild(renderKvRow('Benchmark', ranking.benchmark.name
    + (ranking.benchmark.fallback_used ? ' (fallback — category benchmark had insufficient history)' : '')));
  meta.appendChild(renderKvRow('Funds ranked', ranking.ranked_count + ' of ' + ranking.total_funds_in_category));
  if (ranking.excluded_count > 0) {
    meta.appendChild(renderKvRow('Excluded', ranking.excluded_count + ' (insufficient data)'));
  }
  meta.appendChild(renderKvRow('Computed at', ranking.computed_at));
  header.appendChild(meta);

  /* Limitations */
  const lims = renderLimitations(ranking.limitations, 'Method notes');
  if (lims) header.appendChild(lims);
  insightsEl.appendChild(header);

  /* Ranked fund cards */
  ranking.ranked.forEach((rf) => {
    const card = el('article', 'insight ranking-card');

    /* Rank badge + name */
    const titleRow = el('div', 'rank-title-row');
    const badge = el('span', 'rank-badge', '#' + rf.rank);
    if (rf.rank <= 3) badge.classList.add('rank-top');
    titleRow.appendChild(badge);
    const nameBlock = el('div', 'rank-name-block');
    nameBlock.appendChild(el('span', 'rank-fund-name', rf.fund_name));
    nameBlock.appendChild(el('span', 'rank-fund-house', rf.fund_house));
    titleRow.appendChild(nameBlock);
    const domBadge = el('span', 'rank-dominance', 'Beats ' + rf.dominance.beats + '/' + rf.dominance.of + ' peers');
    titleRow.appendChild(domBadge);
    /* Confidence badge */
    const confClass = rf.confidence_level === 'High' ? 'conf-high' : rf.confidence_level === 'Medium' ? 'conf-med' : 'conf-low';
    const confBadge = el('span', 'rank-confidence ' + confClass, rf.confidence_level + ' (' + rf.history_years + 'y)');
    titleRow.appendChild(confBadge);
    card.appendChild(titleRow);

    /* Metric chips */
    const chips = el('div', 'summary-chips');
    const m = rf.metrics;
    chips.appendChild(summaryChip('Fund CAGR', fmtPct(m.fund_cagr_pct)));
    chips.appendChild(summaryChip('Excess return', fmtDiff(m.excess_return_pct),
      m.excess_return_pct >= 0 ? 'positive' : 'negative'));
    chips.appendChild(summaryChip('Max drawdown', fmtPct(m.max_drawdown_pct), 'neutral'));
    chips.appendChild(summaryChip('Consistency', fmtPct(m.consistency_pct),
      m.consistency_pct >= 50 ? 'positive' : 'negative'));
    chips.appendChild(summaryChip('Volatility', fmtPct(m.volatility_pct)));
    chips.appendChild(summaryChip('Downside capture', m.downside_capture_ratio.toFixed(2),
      m.downside_capture_ratio <= 1.0 ? 'positive' : 'negative'));
    card.appendChild(chips);

    /* Strengths */
    if (rf.strengths.length) {
      const sList = el('ul', 'rank-strengths');
      rf.strengths.forEach((s) => sList.appendChild(el('li', '', s)));
      card.appendChild(sList);
    }

    /* Weaknesses */
    if (rf.weaknesses.length) {
      const wList = el('ul', 'rank-weaknesses');
      rf.weaknesses.forEach((w) => wList.appendChild(el('li', '', w)));
      card.appendChild(wList);
    }

    insightsEl.appendChild(card);
  });

  /* Excluded funds (collapsed) */
  if (ranking.excluded.length) {
    const excSection = el('article', 'insight ranking-excluded');
    const toggle = el('button', 'btn-secondary ranking-excl-toggle',
      'Show ' + ranking.excluded.length + ' excluded fund(s)');
    const excList = el('div', 'ranking-excl-list');
    excList.style.display = 'none';
    ranking.excluded.forEach((ex) => {
      const row = el('div', 'ranking-excl-row');
      row.appendChild(el('span', 'ranking-excl-name', ex.fund_name));
      row.appendChild(el('span', 'ranking-excl-reason', ex.reason));
      excList.appendChild(row);
    });
    toggle.addEventListener('click', () => {
      const open = excList.style.display !== 'none';
      excList.style.display = open ? 'none' : 'block';
      toggle.textContent = (open ? 'Show' : 'Hide') + ' ' + ranking.excluded.length + ' excluded fund(s)';
    });
    excSection.appendChild(toggle);
    excSection.appendChild(excList);
    insightsEl.appendChild(excSection);
  }
}


/* ═══════════════════════════════════════════════════════════════
   Multi-Category Ranking — top picks across all categories
   ═══════════════════════════════════════════════════════════════ */

const multiRankBtn = document.getElementById('multiRankBtn');

multiRankBtn.addEventListener('click', runMultiCategoryRank);

function runMultiCategoryRank() {
  rankResultsEl.innerHTML = '';
  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Ranking all core categories…';
  multiRankBtn.disabled = true;
  rankRunBtn.disabled = true;

  postJson('/analytics/rank-all-categories', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    top_n: 5,
  }).then(function(result) {
    gateStatus.textContent = result.gate.reason;
    renderMultiCategoryResult(result.multi_ranking);
  }).catch(function(error) {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  }).finally(function() {
    multiRankBtn.disabled = false;
    rankRunBtn.disabled = !rankCategorySelect.value;
  });
}

var CATEGORY_SHORT_NAMES = {
  'Equity Scheme - Large Cap Fund': 'Large Cap',
  'Equity Scheme - Mid Cap Fund': 'Mid Cap',
  'Equity Scheme - Small Cap Fund': 'Small Cap',
  'Equity Scheme - Large & Mid Cap Fund': 'Large & Mid Cap',
  'Equity Scheme - Flexi Cap Fund': 'Flexi Cap',
  'Equity Scheme - Multi Cap Fund': 'Multi Cap',
  'Equity Scheme - ELSS': 'ELSS',
  'Equity Scheme - Value Fund': 'Value',
  'Equity Scheme - Focused Fund': 'Focused',
  'Equity Scheme - Dividend Yield Fund': 'Dividend Yield',
  'Equity Scheme - Contra': 'Contra',
};

function renderMultiCategoryResult(multi) {
  if (!multi) return;
  insightsEl.innerHTML = '';

  /* Overall header */
  var header = el('article', 'insight ranking-header');
  header.appendChild(el('h2', 'insight-title', 'Selection View — Top Picks Across Categories'));

  var meta = el('div', 'ranking-meta');
  meta.appendChild(renderKvRow('Categories ranked', multi.categories_ranked.toString()));
  if (multi.categories_failed > 0) {
    meta.appendChild(renderKvRow('Categories failed', multi.categories_failed.toString()));
  }
  meta.appendChild(renderKvRow('Showing', 'Top ' + multi.top_n + ' per category'));
  meta.appendChild(renderKvRow('Computed at', multi.computed_at));
  header.appendChild(meta);

  var lims = renderLimitations([
    'Each category is ranked independently — no cross-category comparison.',
    'Rankings based on pairwise dominance across 5 metrics.',
    'Historical data only — not predictive.',
  ], 'Method notes');
  if (lims) header.appendChild(lims);
  insightsEl.appendChild(header);

  /* Category order: show ranked categories first in a meaningful order */
  var catOrder = [
    'Equity Scheme - Large Cap Fund',
    'Equity Scheme - Mid Cap Fund',
    'Equity Scheme - Small Cap Fund',
    'Equity Scheme - Large & Mid Cap Fund',
    'Equity Scheme - Flexi Cap Fund',
    'Equity Scheme - Multi Cap Fund',
    'Equity Scheme - ELSS',
    'Equity Scheme - Value Fund',
    'Equity Scheme - Focused Fund',
    'Equity Scheme - Dividend Yield Fund',
    'Equity Scheme - Contra',
  ];

  catOrder.forEach(function(cat) {
    var catData = multi.categories[cat];
    if (!catData) return;

    var shortName = CATEGORY_SHORT_NAMES[cat] || cat.replace('Equity Scheme - ', '');

    /* Category section */
    var section = el('article', 'insight multi-cat-section');

    /* Section header */
    var catHeader = el('div', 'multi-cat-header');
    catHeader.appendChild(el('h3', 'multi-cat-title', shortName));
    var benchText = catData.benchmark.name;
    if (catData.benchmark.fallback_used) benchText += ' (fallback)';
    catHeader.appendChild(el('span', 'multi-cat-bench', 'Benchmark: ' + benchText));
    catHeader.appendChild(el('span', 'multi-cat-count',
      catData.ranked_count + ' ranked of ' + catData.total_funds_in_category));
    section.appendChild(catHeader);

    /* Top N fund rows */
    var table = el('div', 'multi-cat-table');
    catData.ranked.forEach(function(rf) {
      var row = el('div', 'multi-cat-row');

      /* Rank */
      var rankBadge = el('span', 'rank-badge rank-badge-sm', '#' + rf.rank);
      if (rf.rank <= 3) rankBadge.classList.add('rank-top');
      row.appendChild(rankBadge);

      /* Fund name + house */
      var nameCol = el('div', 'multi-cat-name-col');
      nameCol.appendChild(el('span', 'rank-fund-name', rf.fund_name));
      nameCol.appendChild(el('span', 'rank-fund-house', rf.fund_house));
      row.appendChild(nameCol);

      /* Key metrics */
      var metricsCol = el('div', 'multi-cat-metrics');
      var m = rf.metrics;
      metricsCol.appendChild(summaryChip('Excess', fmtDiff(m.excess_return_pct),
        m.excess_return_pct >= 0 ? 'positive' : 'negative'));
      metricsCol.appendChild(summaryChip('Consistency', fmtPct(m.consistency_pct),
        m.consistency_pct >= 50 ? 'positive' : 'negative'));
      metricsCol.appendChild(summaryChip('Drawdown', fmtPct(m.max_drawdown_pct), 'neutral'));
      row.appendChild(metricsCol);

      /* Dominance + confidence */
      var rightCol = el('div', 'multi-cat-right');
      rightCol.appendChild(el('span', 'rank-dominance rank-dominance-sm',
        'Beats ' + rf.dominance.beats + '/' + rf.dominance.of));
      var confClass = rf.confidence_level === 'High' ? 'conf-high' : rf.confidence_level === 'Medium' ? 'conf-med' : 'conf-low';
      rightCol.appendChild(el('span', 'rank-confidence ' + confClass,
        rf.confidence_level + ' (' + rf.history_years + 'y)'));
      row.appendChild(rightCol);

      table.appendChild(row);
    });
    section.appendChild(table);
    insightsEl.appendChild(section);
  });

  /* Show errors if any */
  var errorKeys = Object.keys(multi.errors);
  if (errorKeys.length) {
    var errSection = el('article', 'insight ranking-excluded');
    errSection.appendChild(el('h3', 'multi-cat-title', 'Categories Not Ranked'));
    errorKeys.forEach(function(cat) {
      var row = el('div', 'ranking-excl-row');
      var shortName = CATEGORY_SHORT_NAMES[cat] || cat;
      row.appendChild(el('span', 'ranking-excl-name', shortName));
      row.appendChild(el('span', 'ranking-excl-reason', multi.errors[cat]));
      errSection.appendChild(row);
    });
    insightsEl.appendChild(errSection);
  }
}


/* ═══════════════════════════════════════════════════════════════
   Full Investment View — Equity + Debt + Gold (all independent)
   ═══════════════════════════════════════════════════════════════ */

var allAssetsBtn = document.getElementById('allAssetsBtn');

allAssetsBtn.addEventListener('click', runAllAssetsRank);

var DEBT_SHORT_NAMES = {
  'Debt Scheme - Liquid Fund': 'Liquid',
  'Debt Scheme - Ultra Short Duration Fund': 'Ultra Short Duration',
  'Debt Scheme - Short Duration Fund': 'Short Duration',
  'Debt Scheme - Corporate Bond Fund': 'Corporate Bond',
  'Debt Scheme - Banking and PSU Fund': 'Banking & PSU',
  'Debt Scheme - Gilt Fund': 'Gilt',
  'Debt Scheme - Dynamic Bond': 'Dynamic Bond',
  'Debt Scheme - Money Market Fund': 'Money Market',
  'Debt Scheme - Low Duration Fund': 'Low Duration',
  'Debt Scheme - Medium Duration Fund': 'Medium Duration',
  'Debt Scheme - Overnight Fund': 'Overnight',
  'Debt Scheme - Floater Fund': 'Floater',
  'Debt Scheme - Credit Risk Fund': 'Credit Risk',
};

function runAllAssetsRank() {
  rankResultsEl.innerHTML = '';
  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Ranking all asset classes…';
  allAssetsBtn.disabled = true;
  multiRankBtn.disabled = true;
  rankRunBtn.disabled = true;

  postJson('/analytics/rank-all-assets', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    top_n: 3,
  }).then(function(result) {
    gateStatus.textContent = result.gate.reason;
    renderAllAssetsResult(result.all_assets);
  }).catch(function(error) {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  }).finally(function() {
    allAssetsBtn.disabled = false;
    multiRankBtn.disabled = false;
    rankRunBtn.disabled = !rankCategorySelect.value;
  });
}

function renderAllAssetsResult(data) {
  if (!data) return;
  insightsEl.innerHTML = '';

  /* Main header */
  var header = el('article', 'insight ranking-header all-assets-header');
  header.appendChild(el('p', 'global-disclaimer', 'This tool supports decision-making, not guarantees outcomes. Verify independently before acting.'));
  header.appendChild(el('h2', 'insight-title', 'Full Investment View'));

  var meta = el('div', 'ranking-meta');
  meta.appendChild(renderKvRow('Equity categories', data.equity.categories_ranked.toString()));
  meta.appendChild(renderKvRow('Debt categories', data.debt.categories_ranked.toString()));
  meta.appendChild(renderKvRow('Gold', data.gold ? 'Ranked' : (data.gold_error || 'Not available')));
  meta.appendChild(renderKvRow('Showing', 'Top ' + data.top_n + ' per category'));
  meta.appendChild(renderKvRow('Computed at', data.computed_at));
  header.appendChild(meta);

  var lims = renderLimitations([
    'Each category ranked independently — NO cross-asset or cross-category comparison.',
    'Equity uses benchmark-relative metrics. Debt and Gold use absolute metrics.',
    'Historical data only — not predictive.',
    'Results only include currently active funds. Closed/merged funds are excluded (survivorship bias).',
  ], 'Method notes');
  if (lims) header.appendChild(lims);
  insightsEl.appendChild(header);

  /* ── Summary Block — Top candidates (excludes Low confidence) ── */
  if (data.summary) renderSummaryBlock(data.summary);

  /* ── Equity Section ── */
  var eqHeader = el('div', 'asset-class-header');
  eqHeader.appendChild(el('h2', 'asset-class-title', 'Equity'));
  eqHeader.appendChild(el('span', 'asset-class-subtitle', data.equity.categories_ranked + ' categories ranked'));
  insightsEl.appendChild(eqHeader);

  var eqOrder = [
    'Equity Scheme - Large Cap Fund',
    'Equity Scheme - Mid Cap Fund',
    'Equity Scheme - Small Cap Fund',
    'Equity Scheme - Large & Mid Cap Fund',
    'Equity Scheme - Flexi Cap Fund',
    'Equity Scheme - Multi Cap Fund',
  ];

  eqOrder.forEach(function(cat) {
    var catData = data.equity.categories[cat];
    if (!catData) return;
    renderAssetCategorySection(cat, catData, CATEGORY_SHORT_NAMES, 'equity');
  });

  /* Equity errors */
  renderAssetErrors(data.equity.errors, CATEGORY_SHORT_NAMES);

  /* ── Debt Section ── */
  var dtHeader = el('div', 'asset-class-header');
  dtHeader.appendChild(el('h2', 'asset-class-title', 'Debt'));
  dtHeader.appendChild(el('span', 'asset-class-subtitle', data.debt.categories_ranked + ' categories ranked'));
  insightsEl.appendChild(dtHeader);
  insightsEl.appendChild(el('p', 'asset-class-disclaimer', 'Debt returns are not comparable to equity returns. Different risk profiles, tax treatment, and return expectations.'));

  var dtOrder = [
    'Debt Scheme - Short Duration Fund',
    'Debt Scheme - Corporate Bond Fund',
    'Debt Scheme - Banking and PSU Fund',
    'Debt Scheme - Gilt Fund',
    'Debt Scheme - Liquid Fund',
  ];

  dtOrder.forEach(function(cat) {
    var catData = data.debt.categories[cat];
    if (!catData) return;
    renderAssetCategorySection(cat, catData, DEBT_SHORT_NAMES, 'debt');
  });

  renderAssetErrors(data.debt.errors, DEBT_SHORT_NAMES);

  /* ── Gold Section (single best fund) ── */
  if (data.gold && data.gold.ranked && data.gold.ranked.length) {
    var goldHeader = el('div', 'asset-class-header');
    goldHeader.appendChild(el('h2', 'asset-class-title', 'Gold'));
    goldHeader.appendChild(el('span', 'asset-class-subtitle', 'Best gold FoF of ' + data.gold.ranked_count + ' ranked'));
    insightsEl.appendChild(goldHeader);

    renderAssetCategorySection('Gold Fund (FoF)', data.gold, {'Gold Fund (FoF)': 'Gold FoF'}, 'gold');
    var etfNote = el('p', 'gold-etf-note', '\u26A0 Gold ETFs are not supported yet \u2014 only Fund-of-Fund (FoF) wrappers are ranked. Gold FoF invests in Gold ETFs (wrapper structure with additional expense ratio).');
    insightsEl.appendChild(etfNote);
  } else if (data.gold_error) {
    var goldErr = el('article', 'insight ranking-excluded');
    goldErr.appendChild(el('h3', 'multi-cat-title', 'Gold'));
    var errRow = el('div', 'ranking-excl-row');
    errRow.appendChild(el('span', 'ranking-excl-reason', data.gold_error));
    goldErr.appendChild(errRow);
    insightsEl.appendChild(goldErr);
  }

  /* ── Not Supported ── */
  var nsSection = el('article', 'insight ranking-excluded');
  nsSection.appendChild(el('h3', 'multi-cat-title', 'Not Supported'));
  var nsItems = [
    { name: 'REIT / InvIT', reason: 'No live data source available.' },
    { name: 'Fixed Deposits', reason: 'No live data source available.' },
    { name: 'Gold ETF', reason: 'No standardized NAV data in Direct Growth format.' },
  ];
  nsItems.forEach(function(item) {
    var row = el('div', 'ranking-excl-row');
    row.appendChild(el('span', 'ranking-excl-name', item.name));
    row.appendChild(el('span', 'ranking-excl-reason', item.reason));
    nsSection.appendChild(row);
  });
  insightsEl.appendChild(nsSection);
}

function renderAssetCategorySection(cat, catData, nameMap, assetClass) {
  var shortName = nameMap[cat] || cat.replace('Equity Scheme - ', '').replace('Debt Scheme - ', '');
  var section = el('article', 'insight multi-cat-section asset-' + assetClass);

  var catHeader = el('div', 'multi-cat-header');
  catHeader.appendChild(el('h3', 'multi-cat-title', shortName));
  if (catData.benchmark && catData.benchmark.name !== 'None (absolute metrics)') {
    var benchText = catData.benchmark.name;
    if (catData.benchmark.fallback_used) {
      var benchWarn = el('span', 'multi-cat-bench bench-fallback', '\u26A0 Approximate benchmark: ' + benchText);
      catHeader.appendChild(benchWarn);
    } else {
      catHeader.appendChild(el('span', 'multi-cat-bench', 'Benchmark: ' + benchText));
    }
  } else if (assetClass === 'debt' || assetClass === 'gold') {
    catHeader.appendChild(el('span', 'multi-cat-bench', 'Absolute metrics (no benchmark)'));
  }
  if (catData.risk_tag) {
    var riskCls = catData.risk_tag === 'Low' ? 'risk-low' : catData.risk_tag === 'High' ? 'risk-high' : 'risk-med';
    catHeader.appendChild(el('span', 'risk-tag ' + riskCls, 'Risk: ' + catData.risk_tag));
  }
  catHeader.appendChild(el('span', 'multi-cat-count',
    catData.ranked_count + ' ranked of ' + catData.total_funds_in_category));
  section.appendChild(catHeader);

  /* Separate high/med vs low confidence funds */
  var highMedFunds = catData.ranked.filter(function(rf) { return rf.confidence_level !== 'Low'; });
  var lowFunds = catData.ranked.filter(function(rf) { return rf.confidence_level === 'Low'; });

  var table = el('div', 'multi-cat-table');
  highMedFunds.forEach(function(rf) {
    table.appendChild(_buildFundRow(rf, assetClass));
  });

  if (lowFunds.length) {
    var toggleId = 'low-conf-' + cat.replace(/[^a-zA-Z0-9]/g, '-');
    var toggleBtn = el('button', 'low-conf-toggle');
    toggleBtn.textContent = 'Show ' + lowFunds.length + ' low-confidence fund' + (lowFunds.length > 1 ? 's' : '');
    toggleBtn.setAttribute('aria-expanded', 'false');
    var lowContainer = el('div', 'low-conf-container');
    lowContainer.id = toggleId;
    lowContainer.style.display = 'none';
    var lowBanner = el('div', 'low-conf-banner', '\u26A0 Low-confidence data \u2014 limited history, use cautiously. These rankings may not reflect full market cycles.');
    lowContainer.appendChild(lowBanner);
    lowFunds.forEach(function(rf) {
      lowContainer.appendChild(_buildFundRow(rf, assetClass));
    });
    toggleBtn.addEventListener('click', function() {
      var open = lowContainer.style.display !== 'none';
      lowContainer.style.display = open ? 'none' : 'block';
      toggleBtn.textContent = (open ? 'Show ' : 'Hide ') + lowFunds.length + ' low-confidence fund' + (lowFunds.length > 1 ? 's' : '');
      toggleBtn.setAttribute('aria-expanded', String(!open));
    });
    table.appendChild(toggleBtn);
    table.appendChild(lowContainer);
  }

  section.appendChild(table);
  insightsEl.appendChild(section);
}

function _buildFundRow(rf, assetClass) {
  var isLowConf = rf.confidence_level === 'Low';
  var row = el('div', 'multi-cat-row' + (isLowConf ? ' low-confidence-row' : ''));

    var rankBadge = el('span', 'rank-badge rank-badge-sm', '#' + rf.rank);
    if (rf.rank <= 3) rankBadge.classList.add('rank-top');
    if (rf.top_in_category) rankBadge.classList.add('rank-top-cat');
    row.appendChild(rankBadge);

    var nameCol = el('div', 'multi-cat-name-col');
    var nameRow = el('div', 'multi-cat-name-row');
    nameRow.appendChild(el('span', 'rank-fund-name', rf.fund_name));
    if (rf.top_in_category) {
      nameRow.appendChild(el('span', 'top-in-category-label', 'Top in category'));
    }
    nameCol.appendChild(nameRow);
    nameCol.appendChild(el('span', 'rank-fund-house', rf.fund_house));
    row.appendChild(nameCol);

    var metricsCol = el('div', 'multi-cat-metrics');
    var m = rf.metrics;
    if (assetClass === 'equity') {
      metricsCol.appendChild(summaryChip('Excess', fmtDiff(m.excess_return_pct),
        m.excess_return_pct >= 0 ? 'positive' : 'negative'));
      metricsCol.appendChild(summaryChip('Consistency', fmtPct(m.consistency_pct),
        m.consistency_pct >= 50 ? 'positive' : 'negative'));
      metricsCol.appendChild(summaryChip('Drawdown', fmtPct(m.max_drawdown_pct), 'neutral'));
    } else {
      metricsCol.appendChild(summaryChip('CAGR', fmtPct(m.cagr_pct), m.cagr_pct >= 0 ? 'positive' : 'negative'));
      metricsCol.appendChild(summaryChip('Vol', fmtPct(m.volatility_pct), 'neutral'));
      metricsCol.appendChild(summaryChip('Drawdown', fmtPct(m.max_drawdown_pct), 'neutral'));
      metricsCol.appendChild(summaryChip('Risk-Adj', m.risk_adj_return.toFixed(2),
        m.risk_adj_return >= 1 ? 'positive' : 'neutral'));
    }
    row.appendChild(metricsCol);

    var rightCol = el('div', 'multi-cat-right');
    if (assetClass === 'equity') {
      rightCol.appendChild(el('span', 'rank-dominance rank-dominance-sm',
        'Beats ' + rf.dominance.beats + '/' + rf.dominance.of));
    }
    var confClass = rf.confidence_level === 'High' ? 'conf-high' : rf.confidence_level === 'Medium' ? 'conf-med' : 'conf-low';
    rightCol.appendChild(el('span', 'rank-confidence ' + confClass,
      rf.confidence_level + ' (' + rf.history_years + 'y)'));
    row.appendChild(rightCol);

    return row;
}

function renderAssetErrors(errors, nameMap) {
  var keys = Object.keys(errors);
  if (!keys.length) return;
  var section = el('article', 'insight ranking-excluded');
  keys.forEach(function(cat) {
    var row = el('div', 'ranking-excl-row');
    row.appendChild(el('span', 'ranking-excl-name', nameMap[cat] || cat));
    row.appendChild(el('span', 'ranking-excl-reason', errors[cat]));
    section.appendChild(row);
  });
  insightsEl.appendChild(section);
}


/* ═══════════════════════════════════════════════════════════════
   Summary Block — top candidates per category (not a recommendation)
   ═══════════════════════════════════════════════════════════════ */

function renderSummaryBlock(summary) {
  var section = el('article', 'insight summary-block summary-muted');
  section.appendChild(el('h3', 'summary-title-sm', 'Top Candidates per Category'));
  section.appendChild(el('p', 'summary-disclaimer', 'Not a recommendation. Based on historical data only. Low-confidence picks excluded.'));
  section.appendChild(el('p', 'summary-disclaimer', 'Diversify across categories. Do not concentrate all investments in one.'));

  var grid = el('div', 'summary-grid');

  /* Equity picks */
  if (summary.equity && Object.keys(summary.equity).length) {
    var eqCol = el('div', 'summary-col');
    eqCol.appendChild(el('h3', 'summary-col-title', 'Equity'));
    Object.keys(summary.equity).forEach(function(cat) {
      var pick = summary.equity[cat];
      var row = el('div', 'summary-pick-row');
      row.appendChild(el('span', 'summary-pick-cat', cat));
      row.appendChild(el('span', 'summary-pick-arrow', '\u2192'));
      var cleanName = pick.fund_name.replace(/ - Direct.*$/i, '').replace(/ Direct.*$/i, '');
      row.appendChild(el('span', 'summary-pick-name', cleanName));
      var confClass = pick.confidence_level === 'High' ? 'conf-high' : 'conf-med';
      row.appendChild(el('span', 'rank-confidence summary-conf ' + confClass, pick.confidence_level));
      eqCol.appendChild(row);
    });
    grid.appendChild(eqCol);
  }

  /* Debt picks */
  if (summary.debt && Object.keys(summary.debt).length) {
    var dtCol = el('div', 'summary-col');
    dtCol.appendChild(el('h3', 'summary-col-title', 'Debt'));
    Object.keys(summary.debt).forEach(function(cat) {
      var pick = summary.debt[cat];
      var row = el('div', 'summary-pick-row');
      row.appendChild(el('span', 'summary-pick-cat', cat));
      row.appendChild(el('span', 'summary-pick-arrow', '\u2192'));
      var cleanName = pick.fund_name.replace(/ - Direct.*$/i, '').replace(/ Direct.*$/i, '');
      row.appendChild(el('span', 'summary-pick-name', cleanName));
      var confClass = pick.confidence_level === 'High' ? 'conf-high' : 'conf-med';
      row.appendChild(el('span', 'rank-confidence summary-conf ' + confClass, pick.confidence_level));
      dtCol.appendChild(row);
    });
    grid.appendChild(dtCol);
  }

  /* Gold pick */
  if (summary.gold) {
    var otherCol = el('div', 'summary-col');
    otherCol.appendChild(el('h3', 'summary-col-title', 'Gold'));
    var gRow = el('div', 'summary-pick-row');
    gRow.appendChild(el('span', 'summary-pick-cat', 'Gold FoF'));
    gRow.appendChild(el('span', 'summary-pick-arrow', '\u2192'));
    gRow.appendChild(el('span', 'summary-pick-name', summary.gold.fund_name.replace(/ - Direct.*$/i, '').replace(/ Direct.*$/i, '')));
    otherCol.appendChild(gRow);
    grid.appendChild(otherCol);
  }

  section.appendChild(grid);
  insightsEl.appendChild(section);
}


/* ═══════════════════════════════════════════════════════════════
   Portfolio Health Check
   ═══════════════════════════════════════════════════════════════ */

var healthCheckBtn = document.getElementById('healthCheckBtn');
var healthCodesInput = document.getElementById('healthCodesInput');

healthCheckBtn.addEventListener('click', runPortfolioHealth);

/* ═══ CATEGORY CONTEXT (factual SEBI descriptions) ═══ */
var CATEGORY_CONTEXT = {
  'Large Cap': 'Top 100 companies by market cap',
  'Mid Cap': 'Companies ranked 101\u2013250 by market cap',
  'Small Cap': 'Companies ranked 251+ by market cap',
  'Flexi Cap': 'Can invest across market caps',
  'Multi Cap': 'Must invest across large, mid, and small cap',
  'Large & Mid Cap': 'Invests in both large and mid cap segments',
  'ELSS': 'Tax-saving equity fund with 3-year lock-in',
  'Value': 'Value investment strategy',
  'Contra': 'Contrarian investment strategy',
  'Focused': 'Concentrated portfolio (max 30 stocks)',
  'Dividend Yield': 'Invests in high-dividend stocks',
  'Short Duration': 'Debt: 1\u20133 year duration',
  'Corporate Bond': 'Debt: AA+ and above corporate bonds',
  'Liquid': 'Debt: Up to 91-day maturity',
  'Ultra Short Duration': 'Debt: 3\u20136 month duration',
  'Low Duration': 'Debt: 6\u201312 month duration',
  'Medium Duration': 'Debt: 3\u20134 year duration',
  'Medium to Long Duration': 'Debt: 4\u20137 year duration',
  'Long Duration': 'Debt: 7+ year duration',
  'Dynamic Bond': 'Debt: Duration managed dynamically',
  'Banking & PSU': 'Debt: Bank and PSU bonds',
  'Gilt': 'Debt: Government securities',
  'Credit Risk': 'Debt: Below AA rated, higher risk',
  'Overnight': 'Debt: 1-day maturity',
  'Money Market': 'Debt: Up to 1-year maturity',
};

function runPortfolioHealth() {
  var raw = healthCodesInput.value.trim();
  if (!raw) {
    renderError('No input provided. Enter AMFI scheme codes separated by commas (e.g. 120503, 119598).');
    return;
  }

  var parts = raw.split(/[,\s]+/).filter(function(s) { return s.trim().length > 0; });
  var codes = [];
  var invalid = [];
  parts.forEach(function(s) {
    var n = parseInt(s.trim(), 10);
    if (!isNaN(n) && n > 0 && String(n) === s.trim()) {
      codes.push(n);
    } else {
      invalid.push(s.trim());
    }
  });
  if (invalid.length && codes.length) {
    renderError('Invalid codes ignored: ' + invalid.join(', ') + '. Proceeding with ' + codes.length + ' valid code(s).');
  }
  if (!codes.length) {
    renderError('No valid scheme codes found. Enter numeric AMFI codes (e.g. 120503, 119598).');
    return;
  }
  if (codes.length > 20) {
    renderError('Maximum 20 holdings allowed. You entered ' + codes.length + '.');
    return;
  }

  rankResultsEl.innerHTML = '';
  insightsEl.innerHTML = '';
  gateStatus.textContent = 'Checking portfolio health\u2026';
  healthCheckBtn.disabled = true;

  postJson('/analytics/portfolio-health', {
    subject_token: 'demo-user',
    user_country: 'IN',
    asset_market: 'IN',
    serving_entity: 'local_demo',
    scheme_codes: codes,
  }).then(function(result) {
    gateStatus.textContent = result.gate.reason;
    renderPortfolioHealth(result.health);
  }).catch(function(error) {
    gateStatus.textContent = 'Error';
    renderError(error.message);
  }).finally(function() {
    healthCheckBtn.disabled = false;
  });
}

function renderPortfolioHealth(data) {
  if (!data) return;
  insightsEl.innerHTML = '';

  /* All codes not found — abort with clear message */
  if (data.total_holdings === 0 && data.not_found_count > 0) {
    renderError('None of the entered scheme codes were found in the AMFI registry. Please verify the codes and try again.');
    return;
  }

  /* ═══ #7: ONE-LINE FINAL CLARITY (top anchor) ═══ */
  var topClarity = el('p', 'health-top-clarity', 'Use this to review and compare \u2014 not to blindly replace holdings.');
  insightsEl.appendChild(topClarity);

  /* \u2550\u2550\u2550 COVERAGE INTEGRITY BANNER (Item 1) \u2550\u2550\u2550
     If a meaningful share of capital sits in unanalyzable holdings
     (ETFs, hybrids, sectoral, insufficient data), portfolio-level
     conclusions below are based on a partial view. Surface that
     before the user reads any of them. */
  if (data.coverage && data.coverage.confidence_band !== 'full') {
    var covBand = data.coverage.confidence_band; /* "partial" | "low" */
    var covCls = covBand === 'low' ? 'cov-low' : 'cov-partial';
    var covBox = el('article', 'insight health-coverage ' + covCls);
    var covHead = el('div', 'cov-head');
    covHead.appendChild(el('span', 'cov-label',
      covBand === 'low' ? '\u26a0 Limited coverage' : 'Partial coverage'));
    covHead.appendChild(el('span', 'cov-pct',
      data.coverage.analyzed_pct.toFixed(1) + '% analyzed / '
      + data.coverage.not_ranked_pct.toFixed(1) + '% not ranked'));
    covBox.appendChild(covHead);
    covBox.appendChild(el('p', 'cov-note', data.coverage.note));
    insightsEl.appendChild(covBox);
  }

  /* Header */
  var header = el('article', 'insight ranking-header');
  header.appendChild(el('p', 'global-disclaimer', 'This tool supports decision-making, not guarantees outcomes. Verify independently before acting.'));
  header.appendChild(el('h2', 'insight-title', 'Portfolio Health Check'));

  /* ═══ #1: DATA FRESHNESS ═══ */
  if (data.data_as_of) {
    header.appendChild(el('p', 'health-data-freshness', 'Data as of: ' + data.data_as_of + ' \u2014 NAV data may be up to 1 business day delayed from source.'));
  }

  var meta = el('div', 'ranking-meta');
  meta.appendChild(renderKvRow('Holdings checked', data.total_holdings.toString()));
  meta.appendChild(renderKvRow('Not found', data.not_found_count.toString()));
  meta.appendChild(renderKvRow('Computed at', data.computed_at));
  header.appendChild(meta);

  /* ═══ #5: PORTFOLIO STATUS ═══ */
  if (data.portfolio_status) {
    var psCls = data.portfolio_status === 'Highly concentrated' ? 'ps-high' :
                data.portfolio_status === 'Over-diversified' ? 'ps-over' :
                data.portfolio_status === 'Some concentration present' ? 'ps-mod' : 'ps-good';
    var psRow = el('div', 'health-portfolio-status ' + psCls);
    psRow.appendChild(el('span', 'ps-label', 'Portfolio structure:'));
    psRow.appendChild(el('span', 'ps-value', data.portfolio_status));
    header.appendChild(psRow);
  }

  var lims = renderLimitations(data.limitations, 'Method notes');
  if (lims) header.appendChild(lims);
  insightsEl.appendChild(header);

  /* ═══ DECISION SUMMARY (TOP BLOCK) ═══ */
  /* UI-1: each entry carries weight_pct from backend; sorted desc.
     Column header shows count and bucket-level % of portfolio so the
     user can prioritize Reviews by capital impact, not input order. */
  var ds = data.decision_summary;
  var dsWeights = data.decision_summary_weight_pct || {};
  var dsSection = el('article', 'insight decision-summary');
  dsSection.appendChild(el('h3', 'insight-title', 'Decision Summary'));
  dsSection.appendChild(el('p', 'ds-disclaimer', 'Based on peer ranking signals \u2014 not financial advice'));

  var dsGrid = el('div', 'ds-grid');

  var dsGroups = [
    { key: 'Continue', cls: 'ds-continue', icon: '\u2713', label: 'Continue' },
    { key: 'Monitor', cls: 'ds-monitor', icon: '\u25CB', label: 'Monitor' },
    { key: 'Review', cls: 'ds-review', icon: '\u26A0', label: 'Review' },
  ];

  dsGroups.forEach(function(g) {
    var items = ds[g.key] || [];
    if (!items.length) return;
    var col = el('div', 'ds-col ' + g.cls);
    var bucketPct = dsWeights[g.key];
    var headerText = g.icon + ' ' + g.label + ' (' + items.length;
    if (typeof bucketPct === 'number') {
      headerText += ' \u00B7 ' + bucketPct.toFixed(1) + '%';
    }
    headerText += ')';
    col.appendChild(el('h4', 'ds-col-label', headerText));
    items.forEach(function(item) {
      var row = el('div', 'ds-item');
      var name = item.fund_name.replace(/ - Direct.*$/i, '').replace(/ Direct.*$/i, '');
      row.appendChild(el('span', 'ds-item-name', name));
      if (typeof item.weight_pct === 'number') {
        row.appendChild(el('span', 'ds-item-weight', item.weight_pct.toFixed(1) + '%'));
      }
      row.appendChild(el('span', 'ds-item-cat', item.category_short));
      if (item.action_note) {
        row.appendChild(el('span', 'ds-item-note', item.action_note));
      }
      col.appendChild(row);
    });
    dsGrid.appendChild(col);
  });

  dsSection.appendChild(dsGrid);
  /* ═══ BEHAVIORAL WARNING ═══ */
  dsSection.appendChild(el('p', 'health-behavior-warn', '\u26A0 Do not base decisions on a single fund. Review across categories and time horizon.'));
  insightsEl.appendChild(dsSection);

  /* Risk Summary */
  var rs = data.risk_summary;
  var riskSection = el('article', 'insight health-risk-summary');
  riskSection.appendChild(el('h3', 'insight-title', 'Portfolio Risk View'));

  var riskGrid = el('div', 'health-risk-grid');

  var allocCol = el('div', 'health-risk-col');
  allocCol.appendChild(el('h4', 'health-risk-label', 'Allocation'));
  allocCol.appendChild(_riskBar('Equity', rs.equity_pct, 'eq'));
  allocCol.appendChild(_riskBar('Debt', rs.debt_pct, 'dt'));
  if (rs.other_pct > 0) allocCol.appendChild(_riskBar('Other', rs.other_pct, 'other'));
  riskGrid.appendChild(allocCol);

  var statusCol = el('div', 'health-risk-col');
  statusCol.appendChild(el('h4', 'health-risk-label', 'Holdings Status'));
  if (rs.strong_count) statusCol.appendChild(el('span', 'health-status-chip status-strong', rs.strong_count + ' Strong'));
  if (rs.neutral_count) statusCol.appendChild(el('span', 'health-status-chip status-neutral', rs.neutral_count + ' Neutral'));
  if (rs.weak_count) statusCol.appendChild(el('span', 'health-status-chip status-weak', rs.weak_count + ' Weak'));
  if (rs.not_ranked_count) statusCol.appendChild(el('span', 'health-status-chip status-nr', rs.not_ranked_count + ' Not Ranked'));
  riskGrid.appendChild(statusCol);

  var riskCol = el('div', 'health-risk-col');
  riskCol.appendChild(el('h4', 'health-risk-label', 'Risk Level'));
  var riskCls = rs.risk_level === 'High' ? 'risk-high' : rs.risk_level.indexOf('Low') >= 0 ? 'risk-low' : 'risk-med';
  riskCol.appendChild(el('span', 'risk-tag risk-tag-lg ' + riskCls, rs.risk_level));
  if (rs.low_confidence_count > 0) {
    riskCol.appendChild(el('p', 'health-risk-note', rs.low_confidence_count + ' holding(s) have low-confidence data'));
  }
  riskGrid.appendChild(riskCol);

  riskSection.appendChild(riskGrid);
  riskSection.appendChild(el('p', 'health-risk-note', 'Potential overlap between holdings not measured. Actual diversification may differ.'));
  insightsEl.appendChild(riskSection);

  /* ═══ PORTFOLIO ISSUES ═══ */
  var corrPairs = (data.correlations || []);
  var hasIssues = (data.mistakes.length + data.redundancies.length + data.concentration.length + data.exposure_gaps.length + corrPairs.length) > 0;
  if (hasIssues) {
    var issuesSection = el('article', 'insight health-issues');
    issuesSection.appendChild(el('h3', 'insight-title', 'Portfolio Issues'));

    /* Mistakes */
    data.mistakes.forEach(function(m) {
      var row = el('div', 'issue-row issue-' + m.severity);
      row.appendChild(el('span', 'issue-type', m.type.replace(/_/g, ' ')));
      row.appendChild(el('span', 'issue-msg', m.message));
      issuesSection.appendChild(row);
    });

    /* Redundancy */
    data.redundancies.forEach(function(r) {
      var row = el('div', 'issue-row issue-moderate');
      row.appendChild(el('span', 'issue-type', 'redundancy'));
      var nameA = r.fund_a.fund_name.replace(/ - Direct.*$/i, '').replace(/ Direct.*$/i, '');
      var nameB = r.fund_b.fund_name.replace(/ - Direct.*$/i, '').replace(/ Direct.*$/i, '');
      row.appendChild(el('span', 'issue-msg', nameA + ' \u2194 ' + nameB + ' (' + r.category + ') \u2014 ' + r.message));
      issuesSection.appendChild(row);
    });

    /* Concentration */
    data.concentration.forEach(function(c) {
      var row = el('div', 'issue-row issue-moderate');
      row.appendChild(el('span', 'issue-type', 'concentration'));
      row.appendChild(el('span', 'issue-msg', c.message));
      issuesSection.appendChild(row);
    });

    /* Exposure gaps — UI-6 reframe: state observation + diversification
       implication without crossing into advice. "Reduces diversification
       breadth" is a factual property, not a recommendation. */
    data.exposure_gaps.forEach(function(g) {
      var row = el('div', 'issue-row issue-info');
      row.appendChild(el('span', 'issue-type', 'no exposure'));
      var msg = g.message + ' — reduces diversification breadth across this asset class.';
      row.appendChild(el('span', 'issue-msg', msg));
      issuesSection.appendChild(row);
    });

    /* P3: high-correlation pairs (hidden overlap). Different category
       labels can mask the fact that two funds move together — this
       row surfaces the actual return correlation so the user sees
       what category-based diversification logic missed. */
    corrPairs.forEach(function(c) {
      var sevCls = c.correlation >= 0.95 ? 'issue-high' :
                   c.correlation >= 0.90 ? 'issue-moderate' : 'issue-info';
      var row = el('div', 'issue-row ' + sevCls);
      row.appendChild(el('span', 'issue-type', 'high overlap'));
      var nameA = c.fund_a.fund_name.replace(/ - Direct.*$/i, '').replace(/ Direct.*$/i, '');
      var nameB = c.fund_b.fund_name.replace(/ - Direct.*$/i, '').replace(/ Direct.*$/i, '');
      var catNote = c.cross_category
        ? ' [' + c.fund_a.category_short + ' ↔ ' + c.fund_b.category_short + ']'
        : ' [same category]';
      var corrChip = ' (ρ=' + c.correlation.toFixed(2) + ', ' + c.common_days + ' days)';
      row.appendChild(el('span', 'issue-msg',
        nameA + ' ↔ ' + nameB + catNote + corrChip + ' — moves together; actual diversification is lower than category labels suggest.'));
      issuesSection.appendChild(row);
    });

    issuesSection.appendChild(el('p', 'health-risk-note', 'Observations based on historical data, not allocation guidance.'));
    insightsEl.appendChild(issuesSection);
  }

  /* ═══ #5: NO MAJOR ISSUES CONFIRMATION ═══ */
  if (data.no_major_issues) {
    var noIssues = el('article', 'insight health-no-issues');
    noIssues.appendChild(el('p', 'health-no-issues-text', '\u2713 Your portfolio does not show major structural issues based on available data.'));
    insightsEl.appendChild(noIssues);
  }

  /* Per-holding results */
  var holdingsHeader = el('div', 'asset-class-header');
  holdingsHeader.appendChild(el('h2', 'asset-class-title', 'Holdings'));
  holdingsHeader.appendChild(el('span', 'asset-class-subtitle', data.total_holdings + ' fund(s) evaluated'));

  /* ═══ #3: CONFIDENCE FILTER TOGGLE ═══ */
  /* UI-5: defaults ON. Low-confidence results are noisier signals; safer
     to hide by default and let the user opt into seeing them. */
  var confFilterWrap = el('label', 'health-conf-filter');
  var confCheck = document.createElement('input');
  confCheck.type = 'checkbox';
  confCheck.className = 'conf-check';
  confCheck.checked = true;
  confFilterWrap.appendChild(confCheck);
  var lowCount = (data.holdings || []).filter(function(h) {
    return h.confidence_level !== 'High';
  }).length;
  var labelText = ' Show only high-confidence results';
  if (lowCount > 0) {
    labelText += ' (' + lowCount + ' hidden)';
  }
  confFilterWrap.appendChild(document.createTextNode(labelText));
  holdingsHeader.appendChild(confFilterWrap);
  insightsEl.appendChild(holdingsHeader);

  var holdingCards = [];
  data.holdings.forEach(function(h) {
    var card = renderHoldingCard(h);
    if (h.confidence_level !== 'High') {
      card.style.display = 'none';
    }
    holdingCards.push({ card: card, confidence: h.confidence_level });
  });

  confCheck.addEventListener('change', function() {
    holdingCards.forEach(function(item) {
      if (confCheck.checked) {
        item.card.style.display = item.confidence === 'High' ? '' : 'none';
      } else {
        item.card.style.display = '';
      }
    });
  });

  /* Not found */
  if (data.not_found.length) {
    var nfSection = el('article', 'insight ranking-excluded');
    nfSection.appendChild(el('h3', 'multi-cat-title', 'Not Found'));
    data.not_found.forEach(function(nf) {
      var row = el('div', 'ranking-excl-row');
      row.appendChild(el('span', 'ranking-excl-name', 'Code: ' + nf.scheme_code));
      row.appendChild(el('span', 'ranking-excl-reason', nf.reason));
      nfSection.appendChild(row);
    });
    insightsEl.appendChild(nfSection);
  }
}

/* UI-4: map action_note to a concrete next-step the user can take.
   Returns null when the note already explains itself (e.g., the safe-
   peer-fallback "No reliable comparison peer..." string). */
function _notRankedGuidance(h) {
  var note = (h.action_note || '').toLowerCase();
  if (note.indexOf('etf') !== -1) {
    return 'ETF / passive product. Peer-rank logic does not apply. Compare expense ratio and tracking error externally.';
  }
  if (note.indexOf('excluded from ranking') !== -1) {
    return 'Sectoral or thematic fund. Peer comparison across themes is not meaningful — assess against the relevant sector benchmark, not other funds in this list.';
  }
  if (note.indexOf('insufficient data') !== -1 || note.indexOf('could not be ranked') !== -1) {
    if (h.history_years && h.history_years > 0 && h.history_years < 3) {
      return 'Fund history is shorter than 3 years. Re-evaluate after the fund has more data.';
    }
    return 'Not enough comparable peer data to rank. Review the fund’s absolute metrics on its own; a peer comparison will become available as more data accumulates.';
  }
  return null;
}

/* UI-7: side-by-side metric comparison between the held fund and an
   alternative. Renders a compact 3-column grid (Metric / Yours / Alt).
   Only compares the three primary metrics per asset class. Guards on
   missing alt.metrics for backward compatibility. */
function _renderHeldVsAltMetrics(h, a) {
  if (!a || !a.metrics) return null;
  var spec;
  if (h.asset_class === 'equity') {
    spec = [
      { key: 'excess_return_pct', label: 'Excess Ret.', fmt: 'diff' },
      { key: 'consistency_pct',   label: 'Consistency', fmt: 'pct' },
      { key: 'max_drawdown_pct',  label: 'Drawdown',    fmt: 'pct' },
    ];
  } else {
    spec = [
      { key: 'cagr_pct',         label: 'CAGR',     fmt: 'pct' },
      { key: 'volatility_pct',   label: 'Vol',      fmt: 'pct' },
      { key: 'max_drawdown_pct', label: 'Drawdown', fmt: 'pct' },
    ];
  }
  function fmt(v, kind) {
    if (v === null || v === undefined) return '—';
    if (kind === 'diff') return fmtDiff(v);
    return fmtPct(v);
  }
  var grid = el('div', 'health-side-by-side');
  spec.forEach(function(s) {
    var row = el('div', 'sbs-row');
    row.appendChild(el('span', 'sbs-label', s.label));
    row.appendChild(el('span', 'sbs-yours', 'You: ' + fmt(h.metrics ? h.metrics[s.key] : null, s.fmt)));
    row.appendChild(el('span', 'sbs-alt', 'Alt: ' + fmt(a.metrics[s.key], s.fmt)));
    grid.appendChild(row);
  });
  return grid;
}

function _riskBar(label, pct, cls) {
  var wrap = el('div', 'risk-bar-wrap');
  wrap.appendChild(el('span', 'risk-bar-label', label + ': ' + pct + '%'));
  var bar = el('div', 'risk-bar');
  var fill = el('div', 'risk-bar-fill risk-bar-' + cls);
  fill.style.width = Math.min(pct, 100) + '%';
  bar.appendChild(fill);
  wrap.appendChild(bar);
  return wrap;
}

function renderHoldingCard(h) {
  var statusCls = h.status === 'Strong' ? 'status-strong' : h.status === 'Weak' ? 'status-weak' : h.status === 'Neutral' ? 'status-neutral' : 'status-nr';
  var actionCls = h.action === 'Continue' ? 'action-continue' : h.action === 'Review' ? 'action-review' : 'action-monitor';
  var card = el('article', 'insight health-card health-card-' + statusCls);

  /* Header row */
  var cardHeader = el('div', 'health-card-header');

  /* Action badge (primary signal) */
  var actionBadge = el('span', 'health-action-badge ' + actionCls, h.action);
  cardHeader.appendChild(actionBadge);

  var statusLabel = h.status;
  if (h.status === 'Neutral') statusLabel = 'Neutral (monitor)';
  if (h.status === 'Weak') statusLabel = 'Weak (review required)';
  var statusBadge = el('span', 'health-status-badge ' + statusCls, statusLabel);
  cardHeader.appendChild(statusBadge);

  var nameCol = el('div', 'health-card-name');
  nameCol.appendChild(el('span', 'rank-fund-name', h.fund_name));
  nameCol.appendChild(el('span', 'rank-fund-house', h.fund_house));
  cardHeader.appendChild(nameCol);

  var catBadge = el('span', 'health-card-cat', h.category_short);
  cardHeader.appendChild(catBadge);

  /* ═══ #4: CATEGORY CONTEXT ═══ */
  var catCtx = CATEGORY_CONTEXT[h.category_short];
  if (catCtx) {
    cardHeader.appendChild(el('span', 'health-cat-context', catCtx));
  }

  if (h.asset_class === 'debt' && h.risk_tag) {
    var riskCls2 = h.risk_tag === 'Low' ? 'risk-low' : h.risk_tag === 'High' ? 'risk-high' : 'risk-med';
    cardHeader.appendChild(el('span', 'risk-tag ' + riskCls2, 'Risk: ' + h.risk_tag));
  }

  if (h.horizon && h.horizon !== 'Unknown') {
    var hzCls = h.horizon === 'Long-term' ? 'hz-long' : h.horizon === 'Mid-term' ? 'hz-mid' : 'hz-short';
    var hzWrap = el('span', 'health-horizon-wrap');
    hzWrap.appendChild(el('span', 'health-horizon ' + hzCls, h.horizon));
    hzWrap.appendChild(el('span', 'health-horizon-note', 'volatility profile, not a holding-period recommendation'));
    cardHeader.appendChild(hzWrap);
  }

  card.appendChild(cardHeader);

  /* Action note */
  if (h.action_note) {
    card.appendChild(el('p', 'health-action-note', h.action_note));
  }

  /* ═══ UI-4: Not-Ranked / unsupported actionable guidance ═══
     If we couldn't rank this holding, surface what the user can
     actually do instead of leaving them with a bare "Could not be
     ranked" line. */
  if (h.status === 'Not Ranked') {
    var guidance = _notRankedGuidance(h);
    if (guidance) {
      var guideBox = el('div', 'health-not-ranked-guide');
      guideBox.appendChild(el('span', 'health-not-ranked-label', 'What you can do:'));
      guideBox.appendChild(el('span', 'health-not-ranked-text', guidance));
      card.appendChild(guideBox);
    }
  }

  /* ═══ REPEATED BEHAVIORAL WARNING (Review funds only) ═══ */
  if (h.action === 'Review') {
    card.appendChild(el('p', 'health-mini-warn', 'Review across categories and time horizon before acting.'));
  }

  /* Trust block */
  if (h.rank > 0) {
    var trustRow = el('div', 'health-trust-row');
    var confClass = h.confidence_level === 'High' ? 'conf-high' : h.confidence_level === 'Medium' ? 'conf-med' : 'conf-low';
    trustRow.appendChild(el('span', 'trust-item', 'Confidence: '));
    trustRow.appendChild(el('span', 'rank-confidence ' + confClass, h.confidence_level));
    trustRow.appendChild(el('span', 'trust-item', 'History: ' + h.history_years + 'y'));
    if (h.benchmark_name) {
      trustRow.appendChild(el('span', 'trust-item', 'Benchmark: ' + h.benchmark_name));
    }
    trustRow.appendChild(el('span', 'trust-item', 'Rank: ' + h.rank + '/' + h.total_in_category));
    card.appendChild(trustRow);
  }

  /* ═══ DATA QUALITY FLAGS (with severity) ═══ */
  if (h.data_quality_flags && h.data_quality_flags.length) {
    h.data_quality_flags.forEach(function(f) {
      var sevCls = f.severity === 'severe' ? 'dq-severe' : f.severity === 'moderate' ? 'dq-moderate' : 'dq-mild';
      card.appendChild(el('p', 'health-dq-flag ' + sevCls, '\u26A0 ' + f.message));
    });
  }

  /* ═══ OUTLIER FLAGS ═══ */
  if (h.outlier_flags && h.outlier_flags.length) {
    h.outlier_flags.forEach(function(f) {
      card.appendChild(el('p', 'health-outlier-flag', '\u26A0 ' + f));
    });
  }

  /* ═══ YOUR FUND GAPS (personal comparison) ═══ */
  if (h.your_fund_gaps && h.your_fund_gaps.length) {
    var gapsRow = el('div', 'health-your-gaps');
    gapsRow.appendChild(el('span', 'health-your-gaps-label', 'Compared to top-ranked peer, your fund has:'));
    h.your_fund_gaps.forEach(function(g) {
      gapsRow.appendChild(el('span', 'health-your-gap-item', g));
    });
    gapsRow.appendChild(el('span', 'health-your-gaps-caveat', 'Differences are relative within this category \u2014 risk levels differ across categories.'));
    card.appendChild(gapsRow);
  }

  /* Metrics row */
  if (h.rank > 0) {
    var metricsRow = el('div', 'health-rank-row');
    var metricsChips = el('div', 'health-metrics');
    if (h.asset_class === 'equity' && h.metrics.excess_return_pct !== undefined) {
      metricsChips.appendChild(summaryChip('Excess Ret.', fmtDiff(h.metrics.excess_return_pct), 'neutral'));
      metricsChips.appendChild(summaryChip('Consistency', fmtPct(h.metrics.consistency_pct), 'neutral'));
      metricsChips.appendChild(summaryChip('Drawdown', fmtPct(h.metrics.max_drawdown_pct), 'neutral'));
    } else if (h.metrics.cagr_pct !== undefined) {
      metricsChips.appendChild(summaryChip('CAGR', fmtPct(h.metrics.cagr_pct), 'neutral'));
      metricsChips.appendChild(summaryChip('Vol', fmtPct(h.metrics.volatility_pct), 'neutral'));
      metricsChips.appendChild(summaryChip('Drawdown', fmtPct(h.metrics.max_drawdown_pct), 'neutral'));
    }
    metricsRow.appendChild(metricsChips);
    card.appendChild(metricsRow);
  }

  /* Strengths / Weaknesses */
  if (h.strengths.length || h.weaknesses.length) {
    var swRow = el('div', 'health-sw-row');
    h.strengths.forEach(function(s) {
      swRow.appendChild(el('span', 'health-strength', '\u2713 ' + s));
    });
    h.weaknesses.forEach(function(w) {
      swRow.appendChild(el('span', 'health-weakness', '\u2717 ' + w));
    });
    card.appendChild(swRow);
  }

  /* Alternatives (for Weak/Neutral) with justification */
  /* UI-7: each alt now exposes its full metric set (a.metrics) so we
     can render a side-by-side held\u2194alt row for the three primary
     metrics. Justification format upgraded to {reason, magnitude,
     metric} per backend Slice 1. */
  if (h.alternatives.length) {
    var altSection = el('div', 'health-alt-section');
    altSection.appendChild(el('h4', 'health-alt-title', 'Higher-ranked in same category'));
    h.alternatives.forEach(function(a) {
      var altRow = el('div', 'health-alt-row');
      altRow.appendChild(el('span', 'health-alt-rank', '#' + a.rank));
      var altName = a.fund_name.replace(/ - Direct.*$/i, '').replace(/ Direct.*$/i, '');
      altRow.appendChild(el('span', 'health-alt-name', altName));
      var confCls = a.confidence_level === 'High' ? 'conf-high' : 'conf-med';
      altRow.appendChild(el('span', 'rank-confidence ' + confCls, a.confidence_level));
      altSection.appendChild(altRow);
      /* UI-7: side-by-side metrics row */
      var sideBySide = _renderHeldVsAltMetrics(h, a);
      if (sideBySide) altSection.appendChild(sideBySide);
      /* Justification: why this alternative ranks higher (with magnitude) */
      if (a.justification && a.justification.length) {
        var justRow = el('div', 'health-just-row');
        justRow.appendChild(el('span', 'health-just-label', 'Key difference vs your fund:'));
        a.justification.forEach(function(j) {
          var reasonText = (typeof j === 'string') ? j : j.reason;
          var mag = (typeof j === 'object' && j.magnitude) ? j.magnitude : null;
          var item = el('span', 'health-just-item', reasonText);
          if (mag) {
            var magCls = mag === 'large' ? 'mag-large' : mag === 'moderate' ? 'mag-moderate' : 'mag-small';
            item.appendChild(el('span', 'health-just-mag ' + magCls, mag));
          }
          justRow.appendChild(item);
        });
        altSection.appendChild(justRow);
      }
    });
    altSection.appendChild(el('p', 'health-alt-disclaimer', 'Selected from same category ranking. Based on historical data only \u2014 not a recommendation to switch.'));
    altSection.appendChild(el('p', 'health-mini-warn', 'Do not pick a single fund from this list. Compare across your full portfolio.'));
    /* UI-3: switching-cost disclosure. Static observation, no advice. */
    altSection.appendChild(el('p', 'health-switch-cost', 'Switching may incur capital-gains tax (LTCG/STCG depending on holding period and asset class), exit load (typically 1% if redeemed within 1 year for equity), and transaction charges. Confirm specifics with your tax adviser before acting.'));
    card.appendChild(altSection);
  }

  insightsEl.appendChild(card);
  return card;
}
