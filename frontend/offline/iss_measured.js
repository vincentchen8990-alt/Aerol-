/* Orbital Lab — archived ISS measurement mode.
   MARS-S centrifuge RPM replay (MHU-2, 2017) and a separate NASA PIMS acceleration panel (Nov 2026).
   Every value shown is a recorded measurement, a published reference specification, a one-step model
   forecast computed by the backend from fresh *measured* history, or a labelled derived quantity.
   Nothing here is live telemetry, a station simulator or a motor controller. */
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const MICRO = 1e6; // the only conversion applied to PIMS values: displayed micro-g = g x 1e6
  const fx = (v, n) => (Number.isFinite(v) ? v.toFixed(n) : '—');
  const sgn = (v, n) => (Number.isFinite(v) ? (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(n) : '—');
  const unit = (u) => '<span class="unit"> ' + u + '</span>';
  const AXES = ['ax', 'ay', 'az'];
  const AXIS_COLORS = ['#38bdf8', '#4be2a2', '#ffbb49'];
  const STATUS_TEXT = {
    one_step_from_fresh_measured_history: 'one-step forecast from 8 fresh measured samples',
    invalid_negative_rpm: 'INVALID: negative forecast speed (derived g suppressed)',
    insufficient_contiguous_measured_history: 'no forecast: history reset (needs 8 consecutive measured samples)',
    model_unavailable: 'no forecast: model unavailable',
  };

  const S = {
    mode: 'ring', started: false, status: null, statusError: null, hw: null, hwError: null,
    rpm: { data: null, error: null, clip: 0, t: 0, playing: false, speed: 4, cadence: 1 },
    pims: { data: null, error: null, clip: 0, t: 0, playing: false, speed: 16, cadence: 16 },
    factor: 0.1, angle: 0, last: 0,
  };
  window.orbitalLabMeasured = S; // read-only inspection hook for local verification

  // ---- mode switching ---------------------------------------------------------------------------
  function setMode(mode, updateUrl) {
    S.mode = mode === 'iss' ? 'iss' : 'ring';
    $('ring-mode').hidden = S.mode !== 'ring';
    $('iss-mode').hidden = S.mode !== 'iss';
    document.querySelectorAll('[data-mode]').forEach((b) => b.setAttribute('aria-selected', String(b.dataset.mode === S.mode)));
    const tag = $('mode-tag');
    tag.textContent = S.mode === 'iss' ? 'ARCHIVED MEASUREMENTS · NOT LIVE' : 'SIMULATION ONLY';
    tag.classList.toggle('archived', S.mode === 'iss');
    if (updateUrl) history.replaceState(null, '', S.mode === 'iss' ? '#iss' : '#ring');
    if (S.mode === 'iss' && !S.started) start();
    if (S.mode === 'iss') renderAll();
  }

  async function getJSON(url) {
    const r = await fetch(url);
    const d = await r.json().catch(() => ({}));
    if (!r.ok) {
      const x = d && d.detail;
      throw new Error(x && x.message ? x.message : 'HTTP ' + r.status);
    }
    return d;
  }

  async function start() {
    S.started = true;
    renderAll();
    const [st, hw, rpm, pims] = await Promise.allSettled([getJSON('/api/measured/status'), getJSON('/api/measured/hardware'),
      getJSON('/api/measured/rpm/replay'), getJSON('/api/measured/pims/replay')]);
    if (st.status === 'fulfilled') S.status = st.value; else S.statusError = st.reason.message;
    if (hw.status === 'fulfilled') S.hw = hw.value; else S.hwError = hw.reason.message;
    if (rpm.status === 'fulfilled') S.rpm.data = rpm.value; else S.rpm.error = rpm.reason.message;
    if (pims.status === 'fulfilled') S.pims.data = pims.value; else S.pims.error = pims.reason.message;
    fillClips('rpm');
    fillClips('pims');
    renderAll();
  }

  // ---- replay cursor ----------------------------------------------------------------------------
  const clipOf = (k) => (S[k].data && S[k].data.clips ? S[k].data.clips[S[k].clip] : null);
  const clipEnd = (c) => c.points[c.points.length - 1].elapsed_s;

  function cursor(k) {
    const c = clipOf(k);
    if (!c) return null;
    const pts = c.points, t = S[k].t;
    let lo = 0, hi = pts.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (pts[mid].elapsed_s <= t + 1e-9) lo = mid; else hi = mid - 1; }
    // Between two samples further apart than the cadence there is no measurement: a recorded gap.
    return { c, i: lo, p: pts[lo], inGap: t - pts[lo].elapsed_s >= S[k].cadence - 1e-9 };
  }

  function fillClips(k) {
    const sel = $(k + '-clip'), d = S[k].data;
    if (!d) { sel.innerHTML = '<option>unavailable</option>'; sel.disabled = true; return; }
    sel.disabled = false;
    sel.innerHTML = d.clips.map((c, j) => {
      const first = k === 'rpm' ? c.first_timestamp : c.first_timestamp_utc, last = k === 'rpm' ? c.last_timestamp : c.last_timestamp_utc;
      const gaps = c.gaps.length ? ' · ' + c.gaps.length + ' recorded gap' + (c.gaps.length > 1 ? 's' : '') : '';
      return '<option value="' + j + '">' + esc(c.label + ' · ' + first.replace('T', ' ') + ' → ' + last.slice(11) + (k === 'pims' ? ' UTC' : '') + gaps) + '</option>';
    }).join('');
    sel.value = String(S[k].clip);
  }

  function setClip(k, j) { S[k].clip = j; S[k].t = 0; S[k].playing = false; renderAll(); }
  function setTime(k, t) { const c = clipOf(k); if (!c) return; S[k].t = Math.max(0, Math.min(clipEnd(c), t)); renderAll(); }
  function togglePlay(k) {
    const c = clipOf(k); if (!c) return;
    if (!S[k].playing && S[k].t >= clipEnd(c)) S[k].t = 0;
    S[k].playing = !S[k].playing;
    renderAll();
  }

  // ---- MARS-S RPM replay panel ------------------------------------------------------------------
  function renderRpm() {
    const R = S.rpm, d = R.data, n = $('rpm-notice');
    $('rpm-play').textContent = R.playing ? 'Ⅱ Pause' : '▶ Play replay';
    if (!d) {
      n.className = R.error ? 'notice bad' : 'notice';
      n.innerHTML = R.error ? '<strong>MHU-2 replay unavailable:</strong> ' + esc(R.error) + '. No measured values or substitutes are shown.' : 'Loading the archived MHU-2 replay…';
      ['rpm-ts', 'rpm-meas', 'rpm-fc', 'rpm-err', 'rpm-gm', 'rpm-gf'].forEach((x) => { $(x).textContent = '—'; });
      $('rpm-mission').textContent = 'Mission: MHU-2 (unavailable)';
      drawRpmChart();
      return;
    }
    const cur = cursor('rpm'), c = cur.c, p = cur.p, m = d.mission_recorded_span;
    $('rpm-mission').textContent = 'Mission ' + d.mission + ' · flight ' + d.flight_year + (m ? ' · log recorded ' + m.first_timestamp_as_recorded.slice(0, 10) + ' → ' + m.last_timestamp_as_recorded.slice(0, 10) : '');
    $('rpm-timeline').max = String(clipEnd(c));
    $('rpm-timeline').value = String(Math.floor(R.t));
    $('rpm-clock').textContent = 't = ' + Math.floor(R.t) + ' / ' + clipEnd(c) + ' s of recorded clip';
    $('rpm-model-pill').className = 'pill ' + (d.model.available ? 'ok' : 'bad');
    $('rpm-model-pill').textContent = d.model.available ? 'Model: ' + d.model.model_id + ' (hash verified)' : 'Model unavailable';
    let cls = 'notice', html;
    const stamp = new Date(Date.parse(p.timestamp_as_recorded + 'Z') + (cur.inGap ? Math.floor(R.t - p.elapsed_s) * 1000 : 0)).toISOString().slice(0, 19);
    if (cur.inGap) {
      cls += ' warn';
      html = '<strong>Recorded gap at ' + esc(stamp) + ':</strong> no measurement exists for this second in the MHU-2 log. Nothing is interpolated; the next forecast needs 8 new consecutive measured seconds.';
      $('rpm-ts').innerHTML = esc(stamp) + unit('(gap)');
      $('rpm-meas').innerHTML = 'No measurement';
      $('rpm-fc').innerHTML = '—'; $('rpm-err').innerHTML = '—'; $('rpm-gm').innerHTML = '—'; $('rpm-gf').innerHTML = '—';
      $('rpm-history').textContent = 'History resets after a gap; predictions are never used to fill missing seconds.';
    } else {
      $('rpm-ts').innerHTML = esc(p.timestamp_as_recorded) + unit('as recorded');
      $('rpm-meas').innerHTML = fx(p.measured_rpm, 4) + unit('RPM');
      $('rpm-gm').innerHTML = fx(p.derived_nominal_radial_g_from_measured, 4) + unit('g');
      const s = p.forecast_status;
      if (s === 'one_step_from_fresh_measured_history') {
        $('rpm-fc').innerHTML = fx(p.forecast_rpm, 4) + unit('RPM');
        $('rpm-err').innerHTML = sgn(p.forecast_error_rpm, 4) + unit('RPM');
        $('rpm-gf').innerHTML = fx(p.derived_nominal_radial_g_from_forecast, 4) + unit('g');
        html = 'Forecast for <b>' + esc(p.timestamp_as_recorded) + '</b> from the 8 measured samples ' + esc(p.history_start.slice(11)) + '–' + esc(p.forecast_issued_at.slice(11)) + ' (1 s cadence checked). One-step comparison only.';
      } else if (s === 'invalid_negative_rpm') {
        cls += ' bad';
        $('rpm-fc').innerHTML = '<span class="invalid">INVALID</span>' + unit('raw ' + fx(p.raw_model_output_rpm, 4) + ' RPM');
        $('rpm-err').innerHTML = '—';
        $('rpm-gf').innerHTML = '<span class="invalid">suppressed</span>';
        html = '<strong>Invalid forecast:</strong> the model output ' + fx(p.raw_model_output_rpm, 4) + ' RPM, a negative rotor speed outside the measured operating regime. It is shown as invalid, its derived g is suppressed, and the raw unclipped value is kept in the diagnostics.';
      } else {
        cls += s === 'model_unavailable' ? ' bad' : ' warn';
        $('rpm-fc').innerHTML = '<span class="muted">no forecast</span>';
        $('rpm-err').innerHTML = '—';
        $('rpm-gf').innerHTML = '—';
        html = s === 'model_unavailable' ? '<strong>Model unavailable:</strong> ' + esc(d.model.load_error || '') + '. Measured RPM is shown without forecasts.'
          : 'No forecast at ' + esc(p.timestamp_as_recorded) + ': ' + (p.history_reset_reason === 'after_gap' ? 'history reset after a recorded gap' : 'the clip does not contain 8 earlier measured seconds') + '. The model needs 8 consecutive measured seconds.';
      }
      $('rpm-history').textContent = p.forecast_issued_at
        ? 'Model input: measured RPM at ' + p.history_start + ' … ' + p.forecast_issued_at + ' (8 samples, 7 s span) → forecast for ' + p.timestamp_as_recorded + ' (1 s ahead).'
        : 'Model input: — (' + (STATUS_TEXT[s] || s) + ').';
    }
    n.className = cls;
    n.innerHTML = html;
    $('rpm-status').textContent = cur.inGap ? 'recorded gap' : (STATUS_TEXT[p.forecast_status] || p.forecast_status);
    const counts = c.status_counts;
    $('rpm-diag').innerHTML = '<table>' + [
      ['Raw model output at cursor (unclipped)', cur.inGap || p.raw_model_output_rpm == null ? '—' : fx(p.raw_model_output_rpm, 6) + ' RPM'],
      ['Exported forecast (export_rpm_replay.py)', p.exported_forecast_rpm == null ? '— (' + esc(p.exported_forecast_status) + ')' : fx(p.exported_forecast_rpm, 6) + ' RPM'],
      ['Clip statuses', Object.entries(counts).map(([k2, v]) => esc(k2) + ': ' + v).join('<br>')],
      ['Recorded gaps', c.gaps.map((g) => esc(g.after) + ' → ' + esc(g.before) + ' (' + g.missing_samples + ' missing)').join('<br>') || 'none'],
      ['Model file', esc(d.model.file) + (d.model.sha256 ? '<br>sha256 ' + esc(d.model.sha256.slice(0, 16)) + '…' : '')],
      ['Source archive', esc(d.provenance.source_archive) + ' · sha256 ' + (d.provenance.source_archive_present ? (d.provenance.source_archive_sha256_matches ? 'verified locally' : 'MISMATCH') : 'archive not present locally (recorded hash ' + esc((d.provenance.source_sha256_recorded || '').slice(0, 12)) + '…)')],
      ['Timestamps', esc(d.timestamp_timezone)],
    ].map((r) => '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>').join('') + '</table>';
    drawRpmChart();
  }

  function frame(cv) {
    const g = cv.getContext('2d');
    g.clearRect(0, 0, cv.width, cv.height);
    g.fillStyle = '#0a1526';
    g.fillRect(0, 0, cv.width, cv.height);
    g.font = '12px system-ui';
    g.textAlign = 'left';
    return g;
  }

  function emptyChart(g, cv, text) { g.fillStyle = '#8ea6bc'; g.font = '14px system-ui'; g.fillText(text, 24, cv.height / 2); }

  function drawRpmChart() {
    const cv = $('rpm-chart'), g = frame(cv), W = cv.width, H = cv.height, c = clipOf('rpm');
    if (!c) { emptyChart(g, cv, S.rpm.error ? 'MHU-2 replay unavailable — nothing to plot' : 'Loading…'); return; }
    const pts = c.points, L = 64, Rx = W - 14, T = 34, B = H - 30, t1 = clipEnd(c) || 1;
    const X = (t) => L + (Rx - L) * t / t1;
    let lo = 0, hi = 1;
    pts.forEach((p) => { hi = Math.max(hi, p.measured_rpm, p.raw_model_output_rpm ?? -Infinity); lo = Math.min(lo, p.measured_rpm, p.raw_model_output_rpm ?? Infinity); });
    const pad = Math.max((hi - lo) * 0.06, 0.5); lo -= pad; hi += pad;
    const Y = (v) => B - (B - T) * (v - lo) / (hi - lo);
    g.strokeStyle = 'rgba(142,166,188,.18)'; g.lineWidth = 1;
    g.fillStyle = '#b3c7d9'; g.textAlign = 'right';
    for (let j = 0; j <= 4; j++) { const v = lo + (hi - lo) * j / 4, y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(Rx, y); g.stroke(); g.fillText(v.toFixed(1), L - 6, y + 4); }
    g.textAlign = 'left';
    g.setLineDash([3, 4]); g.strokeStyle = 'rgba(255,170,160,.55)'; g.beginPath(); g.moveTo(L, Y(0)); g.lineTo(Rx, Y(0)); g.stroke(); g.setLineDash([]);
    // Recorded gaps: shaded, never bridged.
    pts.forEach((p, i) => {
      if (i && p.missing_samples_before > 0) {
        const a = X(pts[i - 1].elapsed_s), b = X(p.elapsed_s);
        g.fillStyle = 'rgba(255,187,73,.22)'; g.fillRect(a, T, Math.max(3, b - a), B - T);
        g.fillStyle = '#ffcc86'; g.fillText('gap', a + 4, T + 12);
      }
    });
    // History window of the forecast at the cursor (8 measured samples).
    const cur = cursor('rpm');
    if (cur && !cur.inGap && cur.p.forecast_issued_at) {
      const i = cur.i;
      g.fillStyle = 'rgba(244,114,182,.10)';
      g.fillRect(X(pts[i - 8].elapsed_s), T, X(pts[i - 1].elapsed_s) - X(pts[i - 8].elapsed_s), B - T);
    }
    // Measured RPM: segments only between consecutive 1-s samples.
    g.strokeStyle = '#38bdf8'; g.lineWidth = 2.2; g.beginPath();
    pts.forEach((p, i) => { const x = X(p.elapsed_s), y = Y(p.measured_rpm); if (i && p.missing_samples_before === 0) g.lineTo(x, y); else g.moveTo(x, y); });
    g.stroke();
    // One-step forecasts: valid dots; invalid negative outputs as red crosses at the raw value.
    pts.forEach((p) => {
      const x = X(p.elapsed_s);
      if (p.forecast_status === 'one_step_from_fresh_measured_history') {
        g.fillStyle = '#f472b6'; g.beginPath(); g.arc(x, Y(p.forecast_rpm), 2.1, 0, 2 * Math.PI); g.fill();
      } else if (p.forecast_status === 'invalid_negative_rpm') {
        const y = Y(p.raw_model_output_rpm); g.strokeStyle = '#ff6b5e'; g.lineWidth = 1.6;
        g.beginPath(); g.moveTo(x - 3, y - 3); g.lineTo(x + 3, y + 3); g.moveTo(x + 3, y - 3); g.lineTo(x - 3, y + 3); g.stroke();
      } else {
        g.fillStyle = 'rgba(142,166,188,.7)'; g.fillRect(x - 0.5, B - 6, 1.5, 6);
      }
    });
    if (cur) { const x = X(S.rpm.t); g.strokeStyle = '#f8fafc'; g.globalAlpha = 0.6; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, T - 6); g.lineTo(x, B); g.stroke(); g.globalAlpha = 1; }
    g.fillStyle = '#38bdf8'; g.fillRect(L, 12, 18, 3); g.fillStyle = '#d7e5f3'; g.fillText('Measured RPM (1 s cadence)', L + 24, 17);
    g.fillStyle = '#f472b6'; g.beginPath(); g.arc(L + 214, 14, 3, 0, 2 * Math.PI); g.fill(); g.fillStyle = '#d7e5f3'; g.fillText('One-step forecast (RPM)', L + 222, 17);
    g.strokeStyle = '#ff6b5e'; g.beginPath(); g.moveTo(L + 384, 11); g.lineTo(L + 390, 17); g.moveTo(L + 390, 11); g.lineTo(L + 384, 17); g.stroke(); g.fillText('Invalid (negative raw output)', L + 396, 17);
    g.fillStyle = 'rgba(142,166,188,.8)'; g.fillRect(L + 582, 11, 2, 7); g.fillStyle = '#d7e5f3'; g.fillText('No forecast (history reset)', L + 590, 17);
    g.fillStyle = '#b3c7d9'; g.fillText('0 s · ' + c.first_timestamp, L, H - 9); g.textAlign = 'right'; g.fillText(t1 + ' s · ' + c.last_timestamp, Rx, H - 9); g.textAlign = 'left';
  }

  // ---- NASA PIMS panel --------------------------------------------------------------------------
  function renderPims() {
    const R = S.pims, d = R.data, n = $('pims-notice');
    $('pims-play').textContent = R.playing ? 'Ⅱ Pause' : '▶ Play replay';
    if (!d) {
      n.className = R.error ? 'notice bad' : 'notice';
      n.innerHTML = R.error ? '<strong>PIMS replay unavailable:</strong> ' + esc(R.error) + '. No measured values or substitutes are shown.' : 'Loading the archived NASA PIMS replay…';
      $('pims-table').innerHTML = '';
      $('pims-ts').textContent = '—';
      const pm = S.status && S.status.models.iss_pims_acceleration;
      $('pims-model-pill').className = 'pill' + (pm ? (pm.available ? ' ok' : ' bad') : '');
      $('pims-model-pill').textContent = pm ? (pm.available ? 'Model loaded (hash verified) · replay missing' : 'Model unavailable') : 'Model: checking…';
      drawPimsChart();
      return;
    }
    const cur = cursor('pims'), c = cur.c, p = cur.p;
    $('pims-timeline').max = String(clipEnd(c));
    $('pims-timeline').value = String(p.elapsed_s);
    $('pims-clock').textContent = 'sample ' + (cur.i + 1) + ' / ' + c.points.length + ' · split: ' + p.split;
    $('pims-model-pill').className = 'pill ' + (d.model.available ? 'ok' : 'bad');
    $('pims-model-pill').textContent = d.model.available ? 'Model: ' + d.model.model_id + ' (hash verified)' : 'Model unavailable';
    $('pims-ts').innerHTML = esc(p.timestamp_utc) + unit('UTC');
    const ok = p.forecast_status === 'one_step_from_fresh_measured_history';
    $('pims-table').innerHTML = '<tr><th>Axis (' + esc(d.coordinate_frame || 'SSAnalysis') + ')</th><th>Measured</th><th>One-step forecast</th><th>Forecast − measured</th></tr>' + AXES.map((a, j) =>
      '<tr><td style="color:' + AXIS_COLORS[j] + '">' + a + '</td><td>' + fx(p.measured_g[j] * MICRO, 4) + ' µg</td><td>' + (ok ? fx(p.forecast_g[j] * MICRO, 4) + ' µg' : '—') + '</td><td>' + (ok ? sgn(p.forecast_error_g[j] * MICRO, 4) + ' µg' : '—') + '</td></tr>').join('');
    let cls = 'notice', html;
    if (ok) {
      html = 'Forecast for <b>' + esc(p.timestamp_utc) + '</b> from 8 measured samples ' + esc(p.history_start_utc.slice(11)) + '–' + esc(p.forecast_issued_at_utc.slice(11)) + ' UTC (16 s apart, 112 s span; cadence checked). One-step comparison only.';
    } else if (p.forecast_status === 'model_unavailable') {
      cls += ' bad'; html = '<strong>Model unavailable:</strong> ' + esc(d.model.load_error || '') + '. Measured values are shown without forecasts.';
    } else {
      cls += ' warn'; html = 'No forecast at ' + esc(p.timestamp_utc) + ': ' + (p.history_reset_reason === 'after_gap' ? 'history reset after a recorded gap of ' + (c.gaps.map((g) => g.missing_samples).join(', ')) + ' missing 16-s samples' : 'the clip does not contain 8 earlier measured samples') + '.';
    }
    if (p.missing_samples_before > 0) { cls = 'notice warn'; html = '<strong>First sample after a recorded gap</strong> (' + p.missing_samples_before + ' missing 16-s samples, not interpolated). ' + html; }
    n.className = cls;
    n.innerHTML = html;
    $('pims-status').textContent = STATUS_TEXT[p.forecast_status] || p.forecast_status;
    drawPimsChart();
  }

  function drawPimsChart() {
    const cv = $('pims-chart'), g = frame(cv), W = cv.width, H = cv.height, c = clipOf('pims');
    if (!c) { emptyChart(g, cv, S.pims.error ? 'PIMS replay unavailable — nothing to plot' : 'Loading…'); return; }
    const pts = c.points, L = 70, Rx = W - 14, T0 = 30, B0 = H - 26, t1 = clipEnd(c) || 1, rowH = (B0 - T0) / 3;
    const X = (t) => L + (Rx - L) * t / t1;
    const cur = cursor('pims');
    pts.forEach((p, i) => {
      if (i && p.missing_samples_before > 0) {
        const a = X(pts[i - 1].elapsed_s), b = X(p.elapsed_s);
        g.fillStyle = 'rgba(255,187,73,.22)'; g.fillRect(a, T0, Math.max(3, b - a), B0 - T0);
        g.fillStyle = '#ffcc86'; g.fillText('gap: ' + p.missing_samples_before + ' samples', a + 4, T0 + 12);
      }
    });
    if (cur && cur.p.forecast_issued_at_utc) {
      const i = cur.i; g.fillStyle = 'rgba(192,132,252,.10)';
      g.fillRect(X(pts[i - 8].elapsed_s), T0, X(pts[i - 1].elapsed_s) - X(pts[i - 8].elapsed_s), B0 - T0);
    }
    AXES.forEach((a, j) => {
      const top = T0 + j * rowH + 12, bot = T0 + (j + 1) * rowH - 12;
      let lo = Infinity, hi = -Infinity;
      pts.forEach((p) => { const v = p.measured_g[j] * MICRO; lo = Math.min(lo, v); hi = Math.max(hi, v); if (p.forecast_g) { const f = p.forecast_g[j] * MICRO; lo = Math.min(lo, f); hi = Math.max(hi, f); } });
      const pad = Math.max((hi - lo) * 0.08, 0.02); lo -= pad; hi += pad;
      const Y = (v) => bot - (bot - top) * (v - lo) / (hi - lo);
      g.strokeStyle = 'rgba(142,166,188,.18)'; g.beginPath(); g.moveTo(L, top); g.lineTo(Rx, top); g.moveTo(L, bot); g.lineTo(Rx, bot); g.stroke();
      g.fillStyle = '#b3c7d9'; g.textAlign = 'right'; g.fillText(hi.toFixed(2), L - 6, top + 4); g.fillText(lo.toFixed(2), L - 6, bot + 3);
      g.fillStyle = AXIS_COLORS[j]; g.fillText(a + ' µg', L - 6, (top + bot) / 2 + 4); g.textAlign = 'left';
      g.strokeStyle = AXIS_COLORS[j]; g.globalAlpha = 0.45; g.lineWidth = 1; g.beginPath();
      pts.forEach((p, i) => { const x = X(p.elapsed_s), y = Y(p.measured_g[j] * MICRO); if (i && p.missing_samples_before === 0) g.lineTo(x, y); else g.moveTo(x, y); });
      g.stroke(); g.globalAlpha = 1;
      pts.forEach((p) => { g.fillStyle = AXIS_COLORS[j]; g.fillRect(X(p.elapsed_s) - 1.5, Y(p.measured_g[j] * MICRO) - 1.5, 3, 3); });
      pts.forEach((p) => { if (p.forecast_g) { g.strokeStyle = '#c084fc'; g.lineWidth = 1.2; g.beginPath(); g.arc(X(p.elapsed_s), Y(p.forecast_g[j] * MICRO), 2.6, 0, 2 * Math.PI); g.stroke(); } });
    });
    if (cur) { const x = X(cur.p.elapsed_s); g.strokeStyle = '#f8fafc'; g.globalAlpha = 0.6; g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, T0 - 6); g.lineTo(x, B0); g.stroke(); g.globalAlpha = 1; }
    g.fillStyle = '#d7e5f3'; g.fillText('■ measured 16-s samples (thin lines only join consecutive samples; no values exist between them)   ○ one-step forecast, 16 s ahead', L, 16);
    g.fillStyle = '#b3c7d9'; g.fillText(c.first_timestamp_utc, L, H - 8); g.textAlign = 'right'; g.fillText(c.last_timestamp_utc + ' (' + t1 + ' s)', Rx, H - 8); g.textAlign = 'left';
  }

  // ---- model status & hardware reference ------------------------------------------------------------
  function renderModels() {
    const el = $('iss-models');
    if (!S.status) { el.innerHTML = S.statusError ? '<div class="notice bad">Measured-model status unavailable: ' + esc(S.statusError) + '</div>' : 'Checking…'; return; }
    const m = S.status.models.mars_s_rpm, p = S.status.models.iss_pims_acceleration, rm = m.metrics, pm = p.metrics;
    const pill = (x) => '<span class="pill ' + (x.available ? 'ok' : 'bad') + '">' + (x.available ? 'loaded · sha256 verified' : 'UNAVAILABLE') + '</span>';
    const rows = (r) => '<table class="kv">' + r.map((x) => '<tr><td>' + x[0] + '</td><td>' + x[1] + '</td></tr>').join('') + '</table>';
    let html = '<h3 class="sub3">MARS-S next-second RPM ' + pill(m) + '</h3>';
    html += m.available ? '' : '<div class="notice bad">' + esc(m.load_error) + '</div>';
    if (rm) {
      const neg = rm.negative_one_step_predictions, tot = rm.test_windows;
      html += rows([
        ['Input → output', '8 measured RPM (1 s cadence, 7 s span) → next RPM (1 s ahead)'],
        ['Test mission', esc(rm.test_mission)],
        ['One-step RMSE', '<b>MLP ' + fx(rm.one_step_rmse.mlp_residual, 6) + '</b> · persistence ' + fx(rm.one_step_rmse.persistence, 6) + ' · moving mean ' + fx(rm.one_step_rmse.moving_mean_8, 6) + ' · Ridge ' + fx(rm.one_step_rmse.ridge_alpha_1, 6) + ' RPM'],
        ['Changing histories (' + rm.changing_history_windows.toLocaleString() + ')', 'MLP ' + fx(rm.changing_history_rmse.mlp_residual, 6) + ' vs persistence ' + fx(rm.changing_history_rmse.persistence, 6) + ' RPM'],
        ['Negative one-step outputs', neg.toLocaleString() + ' of ' + tot.toLocaleString() + ' (' + (100 * neg / tot).toFixed(2) + '%) — shown as invalid'],
        ['Recursive forecasting', '<span class="invalid">FAILED</span> (60 s RMSE ' + fx(rm.recursive_60s_rmse.mlp_residual, 2) + ' vs persistence ' + fx(rm.recursive_60s_rmse.persistence, 2) + ' RPM) — not offered'],
        ['Not', 'a torque-response model, simulator, controller or health model'],
      ]);
    }
    html += '<h3 class="sub3">NASA PIMS next 16-s acceleration ' + pill(p) + '</h3>';
    html += p.available ? '' : '<div class="notice bad">' + esc(p.load_error) + '</div>';
    if (pm) {
      html += rows([
        ['Input → output', '(8, 3) measured g, 16 s apart (112 s span) → next 3-axis sample in g, 16 s ahead'],
        ['Held-out test', esc(pm.test_period_utc[0].slice(0, 10)) + ' → ' + esc(pm.test_period_utc[1].slice(0, 10)) + ' UTC, ' + pm.test_windows.toLocaleString() + ' windows'],
        ['RMSE (µg)', '<b>MLP ' + fx(pm.one_step_rmse_micro_g.mlp_residual, 6) + '</b> · Ridge ' + fx(pm.one_step_rmse_micro_g.ridge_residual, 6) + ' · persistence ' + fx(pm.one_step_rmse_micro_g.persistence, 6)],
        ['MAE (µg)', 'MLP ' + fx(pm.one_step_mae_micro_g.mlp_residual, 6) + ' · <b>Ridge ' + fx(pm.one_step_mae_micro_g.ridge_residual, 6) + '</b> · persistence ' + fx(pm.one_step_mae_micro_g.persistence, 6)],
        ['Comparison', esc(pm.comparison_note)],
        ['Frame', esc(p.coordinate_frame) + (p.coordinate_frame_comment ? ' (' + esc(p.coordinate_frame_comment) + ')' : '')],
      ]);
    }
    html += '<p class="note">' + esc(S.status.separation) + '</p><p class="note">' + esc(S.status.evaluation_note) + '</p>';
    el.innerHTML = html;
  }

  function renderHardware() {
    const el = $('iss-hardware');
    if (!S.hw) { el.innerHTML = S.hwError ? '<div class="notice bad">Hardware profile unavailable: ' + esc(S.hwError) + '. No station geometry is drawn.</div>' : 'Loading…'; return; }
    const iss = S.hw.profiles.iss_host_station, mars = S.hw.profiles.mars_s_mhu1_mhu2;
    const link = (u, t) => '<a href="' + esc(u) + '" target="_blank" rel="noopener">' + esc(t) + '</a>';
    el.innerHTML = '<table class="kv">' + [
      ['Solar-array span', iss.solar_array_wingspan_m + ' m (end to end; a span, not a ring radius)'],
      ['Solar-array longitudinal extent', iss.solar_array_longitudinal_extent_m + ' m'],
      ['Integrated truss length', iss.truss_length_m + ' m'],
      ['Pressurized-module major axis', iss.pressurized_module_major_axis_m + ' m'],
      ['Reference configuration mass', iss.reference_mass_kg.toLocaleString() + ' kg (reference value; not a measured mass on any telemetry date)'],
      ['Source', link(iss.source_url, 'NASA ISS Facts and Figures') + ' (page updated ' + esc(iss.source_page_updated) + ')'],
      ['MARS-S', esc(mars.hardware_type) + ' in ' + esc(mars.host)],
      ['MARS-S nominal radius / setting', mars.nominal_radius_m + ' m / ' + mars.nominal_rotation_setting_rpm + ' RPM'],
      ['Derived nominal radial g at setting', fx(S.hw.derived.mars_s_nominal_setting_radial_g, 4) + ' g (derived, r·ω²/g₀)'],
      ['MARS-S source', link(mars.source_url, 'ibSLS mission overview') + ' · ' + link(mars.measured_data_release, 'measured data release')],
      ['Not measured', esc(mars.missing_measurements.join(', '))],
    ].map((r) => '<tr><td>' + r[0] + '</td><td>' + r[1] + '</td></tr>').join('') + '</table><ul class="caveats">' + iss.caveats.map((x) => '<li>' + esc(x) + '</li>').join('') + '</ul>';
  }

  // ---- ISS schematic plan view with MARS-S inset ---------------------------------------------------
  function dim(g, x1, y1, x2, y2, label, vertical) {
    g.strokeStyle = '#9fb6cc'; g.fillStyle = '#d7e5f3'; g.lineWidth = 1;
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x2, y2); g.stroke();
    const tick = (x, y) => { g.beginPath(); if (vertical) { g.moveTo(x - 5, y); g.lineTo(x + 5, y); } else { g.moveTo(x, y - 5); g.lineTo(x, y + 5); } g.stroke(); };
    tick(x1, y1); tick(x2, y2);
    g.font = '12px system-ui';
    if (vertical) {
      g.save(); g.translate(x1 + 14, (y1 + y2) / 2); g.rotate(-Math.PI / 2); g.textAlign = 'center'; g.fillText(label, 0, 0); g.restore();
    } else { g.textAlign = 'center'; g.fillText(label, (x1 + x2) / 2, y1 - 6); g.textAlign = 'left'; }
  }

  function drawIss() {
    const cv = $('iss-canvas'), g = cv.getContext('2d'), W = cv.width, H = cv.height;
    g.clearRect(0, 0, W, H); g.fillStyle = '#08121f'; g.fillRect(0, 0, W, H); g.font = '13px system-ui'; g.textAlign = 'left';
    if (!S.hw) { g.fillStyle = '#8ea6bc'; g.fillText(S.hwError ? 'Hardware profile unavailable — no geometry is drawn' : 'Loading NASA/JAXA reference dimensions…', 24, H / 2); return; }
    const iss = S.hw.profiles.iss_host_station, mars = S.hw.profiles.mars_s_mhu1_mhu2;
    const span = iss.solar_array_wingspan_m, along = iss.solar_array_longitudinal_extent_m, truss = iss.truss_length_m, mod = iss.pressurized_module_major_axis_m;
    // Plan view: x = truss axis, y = module axis. One scale (px per metre) for every station element.
    const k = Math.min(560 / span, 400 / along), cx = 320, cy = 292;
    const P = (x, y) => [cx + x * k, cy - y * k];
    const rect = (x0, y0, x1, y1, fill, stroke) => { const a = P(x0, y1), b = P(x1, y0); g.fillStyle = fill; g.fillRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1; g.strokeRect(a[0], a[1], b[0] - a[0], b[1] - a[1]); } };
    g.fillStyle = '#9ce8f6'; g.font = 'bold 13px system-ui';
    g.fillText('ISS · SCHEMATIC PLAN VIEW — proportional to NASA reference dimensions, not CAD geometry', 16, 22);
    g.font = '12px system-ui'; g.fillStyle = '#8ea6bc';
    g.fillText('Element widths and positions are illustrative; only the four labelled extents are published values.', 16, 40);
    // Solar arrays: two assemblies per side; outer edges at the published span, blankets spanning the longitudinal extent.
    const w = (span - truss) / 2 + 4.5, half = span / 2, A = along / 2;
    [[half - w, half], [half - 2 * w - 2.5, half - w - 2.5]].forEach(([x0, x1]) => [1, -1].forEach((s) => {
      const lo = Math.min(s * x0, s * x1), hi = Math.max(s * x0, s * x1);
      rect(lo, 1.2, hi, A, '#16406b', '#3b78b0'); rect(lo, -A, hi, -1.2, '#16406b', '#3b78b0');
    }));
    rect(-truss / 2, -1.2, truss / 2, 1.2, '#8193a8', null); // integrated truss
    rect(-2.2, -mod / 2, 2.2, mod / 2, '#c9d4e2', '#e9f0f8'); // pressurized module axis
    const kibo = [-13.5, mod / 2 - 8.5, -2.2, mod / 2 - 4]; // illustrative Kibo position (port side, forward)
    rect(kibo[0], kibo[1], kibo[2], kibo[3], '#ffda4b', '#fff1a8');
    const kp = P(kibo[0], (kibo[1] + kibo[3]) / 2), kb = P(kibo[0], kibo[1]);
    g.fillStyle = '#ffe88b'; g.font = '11px system-ui';
    ['Kibo JPM', '(CBEF / MARS-S host;', 'position schematic)'].forEach((t, j) => g.fillText(t, kb[0] - 64, kb[1] + 14 + j * 13));
    g.font = '12px system-ui';
    // Published extents.
    let a = P(-half, A + 6), b = P(half, A + 6); dim(g, a[0], a[1], b[0], b[1], span + ' m solar-array span (end to end) — a span, not a ring radius', false);
    a = P(-truss / 2, -A - 7); b = P(truss / 2, -A - 7); dim(g, a[0], a[1], b[0], b[1], truss + ' m integrated truss length', false);
    a = P(half + 4, A); b = P(half + 4, -A); dim(g, a[0], a[1], b[0], b[1], along + ' m solar-array longitudinal extent', true);
    a = P(9, mod / 2); b = P(9, -mod / 2); dim(g, a[0], a[1], b[0], b[1], mod + ' m pressurized-module major axis', true);
    a = P(-half, -A - 15); g.fillStyle = '#d7e5f3'; g.fillRect(a[0], a[1], 20 * k, 3); g.fillText('20 m', a[0] + 20 * k + 6, a[1] + 5);
    g.fillStyle = '#8ea6bc'; g.fillText('Reference configuration mass ' + iss.reference_mass_kg.toLocaleString() + ' kg (reference, not measured)', a[0] + 20 * k + 50, a[1] + 5);
    // MARS-S enlarged inset.
    const ix = 652, iy = 64, iw = 292, ih = 456, rc = [ix + iw / 2, iy + 168], Rpx = 104;
    g.strokeStyle = '#2a5266'; g.lineWidth = 1.2; g.strokeRect(ix, iy, iw, ih);
    g.setLineDash([4, 4]); g.beginPath(); g.moveTo(kp[0] + (kibo[2] - kibo[0]) * k, kp[1]); g.lineTo(ix, iy + 40); g.stroke(); g.setLineDash([]);
    g.fillStyle = '#9ce8f6'; g.font = 'bold 13px system-ui'; g.fillText('MARS-S centrifuge — enlarged inset', ix + 10, iy + 20);
    g.font = '12px system-ui'; g.fillStyle = '#8ea6bc';
    const mag = (Rpx / mars.nominal_radius_m) / k;
    g.fillText('Schematic rotor; inset magnified ≈' + Math.round(mag) + '× vs plan view', ix + 10, iy + 38);
    g.strokeStyle = '#3dc5ff'; g.lineWidth = 3; g.beginPath(); g.arc(rc[0], rc[1], Rpx, 0, 2 * Math.PI); g.stroke();
    for (let j = 0; j < 4; j++) {
      const t = S.angle + j * Math.PI / 2;
      g.strokeStyle = '#71859c'; g.lineWidth = 3; g.beginPath(); g.moveTo(rc[0], rc[1]); g.lineTo(rc[0] + Rpx * Math.cos(t), rc[1] + Rpx * Math.sin(t)); g.stroke();
    }
    g.fillStyle = '#a8b8cc'; g.beginPath(); g.arc(rc[0], rc[1], 9, 0, 2 * Math.PI); g.fill();
    const mk = [rc[0] + Rpx * Math.cos(S.angle), rc[1] + Rpx * Math.sin(S.angle)];
    g.fillStyle = '#ffda4b'; g.fillRect(mk[0] - 7, mk[1] - 7, 14, 14);
    g.strokeStyle = '#4be2a2'; g.lineWidth = 2.5; g.beginPath(); g.moveTo(mk[0], mk[1]); g.lineTo(mk[0] + 34 * Math.cos(S.angle), mk[1] + 34 * Math.sin(S.angle)); g.stroke();
    g.strokeStyle = '#d7e5f3'; g.lineWidth = 1; g.beginPath(); g.moveTo(rc[0], rc[1] + Rpx + 16); g.lineTo(rc[0] + Rpx, rc[1] + Rpx + 16); g.stroke();
    g.fillStyle = '#d7e5f3'; g.fillText('r = ' + mars.nominal_radius_m + ' m (nominal)', rc[0] + 6, rc[1] + Rpx + 32);
    const cur = cursor('rpm');
    const lines = [['Nominal: ' + mars.nominal_rotation_setting_rpm + ' RPM setting · r = ' + mars.nominal_radius_m + ' m', '#d7e5f3']];
    if (cur && !cur.inGap) {
      lines.push(['Measured (MHU-2): ' + cur.p.measured_rpm.toFixed(4) + ' RPM', '#d7e5f3']);
      lines.push(['  at ' + cur.p.timestamp_as_recorded.replace('T', ' ') + ' (as recorded)', '#8ea6bc']);
      lines.push(['Derived nominal radial g: ' + fx(cur.p.derived_nominal_radial_g_from_measured, 4) + ' g', '#d7e5f3']);
    } else if (cur) {
      lines.push(['Recorded gap: no measurement;', '#ffcc86'], ['rotor frozen, nothing interpolated', '#ffcc86']);
    } else {
      lines.push([S.rpm.error ? 'Replay unavailable: rotor not animated' : 'Loading replay…', '#ffb4a8']);
    }
    lines.push([S.factor === 1 ? 'Display rotation: true measured rate' : 'Display rotation: ' + (S.factor === 0.25 ? '¼' : '1/' + Math.round(1 / S.factor)) + ' of measured RPM', '#ffcc86']);
    lines.push(['Numbers show the recorded RPM', '#ffcc86']);
    lines.push([S.rpm.playing ? 'Replaying recorded RPM' : 'Paused: press ▶ Play replay to animate', '#8ea6bc']);
    lines.forEach(([t, color], j) => { g.fillStyle = color; g.fillText(t, ix + 10, iy + 322 + j * 18); });
  }

  // ---- render loop --------------------------------------------------------------------------------
  function renderAll() {
    if (S.mode !== 'iss') return;
    renderRpm(); renderPims(); renderModels(); renderHardware(); drawIss();
    const f = S.factor;
    $('iss-display-note').textContent = f === 1 ? 'Rotor animated at the true measured rate.' : 'Rotor animated at ' + (f === 0.25 ? '¼' : '1/' + Math.round(1 / f)) + ' of the measured RPM for visibility; all numbers show the recorded RPM.';
  }

  function tick(now) {
    const dt = S.last ? Math.min(0.1, (now - S.last) / 1000) : 0;
    S.last = now;
    if (S.mode === 'iss') {
      let dirty = false;
      ['rpm', 'pims'].forEach((k) => {
        const R = S[k], c = clipOf(k);
        if (R.playing && c) { R.t = Math.min(clipEnd(c), R.t + dt * R.speed); if (R.t >= clipEnd(c)) R.playing = false; dirty = true; }
      });
      const cur = cursor('rpm');
      // Rotor display rate = measured RPM x display factor, in real time; frozen when paused or in a recorded gap.
      if (S.rpm.playing && cur && !cur.inGap) S.angle = (S.angle + 2 * Math.PI * (cur.p.measured_rpm / 60) * S.factor * dt) % (2 * Math.PI);
      if (dirty) { renderRpm(); renderPims(); }
      drawIss();
    }
    requestAnimationFrame(tick);
  }

  // ---- wiring -------------------------------------------------------------------------------------
  document.querySelectorAll('[data-mode]').forEach((b) => { b.onclick = () => setMode(b.dataset.mode, true); });
  $('iss-display-factor').onchange = (e) => { S.factor = +e.target.value; renderAll(); };
  ['rpm', 'pims'].forEach((k) => {
    $(k + '-clip').onchange = (e) => setClip(k, +e.target.value);
    $(k + '-play').onclick = () => togglePlay(k);
    $(k + '-speed').onchange = (e) => { S[k].speed = +e.target.value; };
    $(k + '-timeline').oninput = (e) => { S[k].playing = false; setTime(k, +e.target.value); };
    $(k + '-back').onclick = () => {
      S[k].playing = false;
      const cur = cursor(k);
      if (cur) setTime(k, cur.inGap || !cur.i ? cur.p.elapsed_s : cur.c.points[cur.i - 1].elapsed_s);
    };
    $(k + '-fwd').onclick = () => { S[k].playing = false; const cur = cursor(k); if (cur && cur.i < cur.c.points.length - 1) setTime(k, cur.c.points[cur.i + 1].elapsed_s); };
  });
  window.addEventListener('hashchange', () => setMode(location.hash === '#iss' ? 'iss' : 'ring', false));
  setMode(location.hash === '#iss' ? 'iss' : 'ring', false);
  requestAnimationFrame(tick);
})();
