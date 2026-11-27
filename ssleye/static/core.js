'use strict';
// data + drawing helpers (window.SE)
(function () {
  const SE = {};
  SE.$ = (s, el = document) => el.querySelector(s);
  SE.$$ = (s, el = document) => [...el.querySelectorAll(s)];
  SE.h = function (tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null) continue;
      if (k === 'class') e.className = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v);
    }
    for (const c of kids) if (c != null) e.append(c);
    return e;
  };
  SE.api = async p => { const r = await fetch(p, { cache: 'no-store' }); if (!r.ok) throw new Error(p + ' ' + r.status); return r.json(); };
  SE.img = src => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });
  SE.css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  SE.LABEL = { loss: 'Loss', grad_norm: 'Gradient norm', img_per_s: 'Images / s', knn_acc: 'kNN accuracy', rankme: 'Effective rank',
    alpha: 'Spectrum decay α', lidar: 'LiDAR', std_ratio: 'Embedding spread', align: 'Alignment', uniform: 'Uniformity', silhouette: 'Cluster quality',
    nbr_stability: 'Neighbour stability', patch_locality: 'Patch locality', gram_drift: 'Patch Gram drift',
    'dino/teacher_entropy': 'Teacher entropy', 'dino/top_proto_share': 'Top prototype share', 'dino/protos_used': 'Prototypes in use',
    'dino/kl': 'Student–teacher KL', dino_loss: 'DINO loss', ibot_loss: 'iBOT patch loss', koleo: 'KoLeo', gram_loss: 'Gram anchoring loss' };
  SE.hasLabels = () => !!(S.probe && S.probe.labels);
  SE.EXPLAIN = {
    loss: 'The SSL objective being minimised. Lower is better, but a low loss alone does not prove the encoder is useful.',
    knn_acc: 'Share of probe images whose 20 nearest neighbours vote for the right class. Uses labels only for scoring.',
    rankme: 'RankMe: how many embedding dimensions carry information. Higher means a richer representation; a fall warns of collapse.',
    alpha: 'α-ReQ: how fast the eigenvalue spectrum decays. Lower means information is spread over more dimensions.',
    lidar: 'LiDAR: effective rank after separating different images from augmented views of the same image. Higher is better.',
    std_ratio: 'Spread of normalised embeddings, scaled so about 1 is healthy and 0 means every image maps to the same point.',
    align: 'How close two augmented views of the same image land. Lower is better.',
    uniform: 'How evenly embeddings cover the sphere. Lower (more negative) is better.',
    silhouette: 'How cleanly k-means clusters separate in embedding space, without labels. Higher is better.',
    grad_norm: 'Size of the gradient each step. Sudden spikes mean unstable training.',
    patch_locality: 'How much more alike neighbouring patches are than random patch pairs in the same image. Falls when patch (dense) features degrade, the failure DINOv3\'s Gram anchoring prevents. No labels needed.',
    gram_drift: 'How far the patch-to-patch similarity pattern has moved from the snapshot where patch locality was best. Rising drift with falling locality = dense features degrading.',
    'dino/teacher_entropy': 'Entropy of the teacher\'s prototype distribution, 0 to 1. Near 1 = flat output (uniform collapse). Near 0 = very sharp.',
    'dino/top_proto_share': 'Share of the batch whose favourite prototype is the single most common one. Near 1 = one-prototype collapse.',
    'dino/protos_used': 'Share of prototypes that are the top choice for at least one image. Very low = the head only uses a few prototypes.',
    'dino/kl': 'How far the student still is from the teacher. Falling to 0 together with a flat teacher means collapse.',
    dino_loss: 'Cross-entropy between teacher and student prototype distributions across views.',
    ibot_loss: 'Masked patch prediction loss (iBOT), used in DINOv2 and DINOv3.',
    koleo: 'KoLeo regulariser: pushes features of different images apart (DINOv2/v3).',
    gram_loss: 'Gram anchoring (DINOv3): keeps patch similarities close to an earlier teacher so dense features do not degrade.',
    nbr_stability: 'Share of each image\'s 20 nearest neighbours that were also its neighbours at the previous snapshot. Near 1 means the representation has settled; low means it is still reorganising. Needs no labels.',
  };
  SE.HIGHER_BETTER = { knn_acc: 1, rankme: 1, lidar: 1, std_ratio: 1, silhouette: 1, loss: -1, alpha: -1, align: -1, uniform: -1 };

  SE.fmt = function (k, v) {
    if (v == null || Number.isNaN(v)) return '—';
    if (k === 'knn_acc' || k === 'margin' || k === 'nbr_stability' || k === 'stable') return (v * 100).toFixed(1) + '%';
    if (k === 'img_per_s' || k === 'batch_size' || k === 'emb_dim' || k === 'rankme' || k === 'lidar' || k === 'step') return Math.round(v).toLocaleString();
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-2 || a >= 1e5)) return v.toExponential(2);
    if (a >= 100) return v.toFixed(0);
    return Number(v.toPrecision(3)).toString();
  };

  const S = SE.S = { base: '', key: '', meta: null, probe: null, scalars: [], snaps: [], alerts: [], stream: [], samples: [], downstream: [], off: {},
    cache: new Map(), sprites: {}, ser: {}, polling: false, listeners: [] };
  SE.on = f => S.listeners.push(f);

  SE.runs = () => SE.api('/api/runs');
  SE.open = async function (key) {
    const [p, ...rest] = key.split('/');
    Object.assign(S, { key, base: `/api/run/${encodeURIComponent(p)}/${encodeURIComponent(rest.join('/'))}`, meta: null, probe: null,
      scalars: [], snaps: [], alerts: [], stream: [], samples: [], downstream: [], off: {}, cache: new Map(), sprites: {}, ser: {} });
    await SE.poll();
  };
  SE.poll = async function () {
    if (!S.base || S.polling) return;
    S.polling = true;
    const base = S.base, kinds = ['scalars', 'snapshots', 'alerts', 'stream', 'samples', 'downstream'];
    try {
      const [meta, ...res] = await Promise.all([SE.api(base + '/meta'),
        ...kinds.map(k => SE.api(`${base}/jsonl/${k}?since=${S.off[k] || 0}`).catch(() => ({ offset: S.off[k] || 0, items: [] })))]);
      if (base !== S.base) return;
      S.meta = meta;
      if (!S.probe && meta.probe) {
        S.probe = meta.probe;
        const [d, t] = await Promise.all([SE.img(`${base}/file/${meta.probe.display.file}`), SE.img(`${base}/file/${meta.probe.thumb.file}`)]);
        S.sprites = { display: d, thumb: t };
        S.spriteFiles = new Map([[meta.probe.display.file, d]]);
      } else if (meta.probe) S.probe = meta.probe;
      const got = { snaps: 0, alerts: 0, stream: 0, scalars: 0, samples: 0, downstream: 0 };
      res.forEach((r, i) => {
        const k = kinds[i], field = k === 'snapshots' ? 'snaps' : k;
        S.off[k] = r.offset;
        if (r.items.length) { S[field].push(...r.items); got[field] = r.items.length; }
      });
      for (const f of S.listeners) f(got);
    } catch (e) { console.warn(e); } finally { S.polling = false; }
  };
  SE.start = async function (onUpdate) {
    if (onUpdate) SE.on(onUpdate);
    const runs = await SE.runs().catch(() => []);
    const want = decodeURIComponent(location.hash.slice(1));
    const key = runs.find(r => r.project + '/' + r.name === want) ? want : (runs[0] ? runs[0].project + '/' + runs[0].name : null);
    if (key) await SE.open(key);
    setInterval(SE.poll, 1500);
    return runs;
  };
  SE.live = () => S.meta && S.meta.status !== 'finished' && S.meta.last_write && (S.meta.now - S.meta.last_write) < 30;
  SE.statusText = () => !S.meta ? '' : S.meta.status === 'finished' ? 'Finished' : SE.live() ? 'Live' : 'Stopped';

  SE.series = function (key) {
    let c = S.ser[key];
    if (!c) c = S.ser[key] = { len: 0, xs: [], ys: [] };
    for (let i = c.len; i < S.scalars.length; i++) { const r = S.scalars[i]; if (key in r) { c.xs.push(r.step); c.ys.push(r[key]); } }
    c.len = S.scalars.length;
    return c;
  };
  SE.snapSeries = key => ({ xs: S.snaps.map(s => s.step), ys: S.snaps.map(s => (s.metrics || {})[key] ?? null) });
  SE.valueAt = function (xs, ys, x) {
    let lo = 0, hi = xs.length - 1, ans = -1;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (xs[m] <= x) { ans = m; lo = m + 1; } else hi = m - 1; }
    for (let i = ans; i >= 0; i--) if (ys[i] != null) return ys[i];
    return null;
  };
  SE.lastStep = () => S.scalars.length ? S.scalars[S.scalars.length - 1].step : 0;
  SE.lastScalar = () => S.scalars[S.scalars.length - 1] || {};
  SE.className = c => (c == null ? '' : S.probe && S.probe.class_names && S.probe.class_names[c] != null ? S.probe.class_names[c] : String(c));

  function b64(s) { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; }
  SE.snap = async function (i) {
    const st = S.snaps[i] && S.snaps[i].step;
    if (st == null) return null;
    if (S.cache.has(st)) return S.cache.get(st);
    const s = await SE.api(`${S.base}/snapshot/${st}`);
    if (s.attn) s.attn.u8 = b64(s.attn.data);
    if (s.tokens) s.tokens.i8 = new Int8Array(b64(s.tokens.data).buffer);
    if (s.layer_attn) { s.layer_attn.u8 = b64(s.layer_attn.data); delete s.layer_attn.data; }
    s.attn && delete s.attn.data; s.tokens && delete s.tokens.data;
    const ver = s.display_version || 0;
    const dv = ((S.probe && S.probe.display_versions) || []).find(x => x.version === ver);
    const file = dv ? dv.file : ver ? `display_sprite_v${ver}.png` : 'display_sprite.png';
    if (!S.spriteFiles) S.spriteFiles = new Map();
    if (!S.spriteFiles.has(file)) S.spriteFiles.set(file, await SE.img(`${S.base}/file/${file}`).catch(() => null));
    s.sprite = S.spriteFiles.get(file);
    S.cache.set(st, s);
    if (S.cache.size > 60) { const k = S.cache.keys().next().value; S.cache.delete(k); }
    return s;
  };
  SE.attnMap = function (s, j, head) {
    if (!s || !s.attn) return null;
    const H = s.attn.heads, P = s.attn.grid ** 2, out = new Float32Array(P);
    const hs = head == null ? [...Array(H).keys()] : [head];
    for (const hh of hs) { const o = (j * H + hh) * P; for (let p = 0; p < P; p++) out[p] += s.attn.u8[o + p] / 255; }
    let mx = 0; for (const v of out) mx = Math.max(mx, v);
    if (mx > 0) for (let p = 0; p < P; p++) out[p] /= mx;
    return out;
  };
  SE.simMap = function (s, jq, pq, j) {
    if (!s || !s.tokens) return null;
    const D = s.tokens.dim, P = s.tokens.grid ** 2, t = s.tokens.i8, out = new Float32Array(P), qo = (jq * P + pq) * D;
    for (let p = 0; p < P; p++) { const o = (j * P + p) * D; let d = 0; for (let k = 0; k < D; k++) d += t[qo + k] * t[o + k]; out[p] = d / 16129; }
    return out;
  };
  SE.peakPatch = function (s, j) {
    const a = SE.attnMap(s, j, null);
    if (a) { let b = 0; for (let p = 1; p < a.length; p++) if (a[p] > a[b]) b = p; return b; }
    const g = s && s.tokens ? s.tokens.grid : 1;
    return Math.floor(g / 2) * g + Math.floor(g / 2);
  };
  SE.grid = s => (s && s.attn ? s.attn.grid : s && s.tokens ? s.tokens.grid : 0);

  SE.fit = function (cv, cssH) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(10, cv.clientWidth || cv.parentElement.clientWidth);
    const hh = cssH == null ? w : cssH;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(hh * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr); cv.style.height = hh + 'px'; }
    const ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [ctx, w, hh];
  };
  SE.drawImg = function (ctx, j, w, hh, snap) {
    const sn = snap || S.view;
    const img = (sn && sn.sprite) || S.sprites.display, s = S.probe.display.size;
    if (!img) return;
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(img, j * s, 0, s, s, 0, 0, w, hh);
  };
  SE.drawThumb = function (ctx, n, x, y, w, hh) {
    const img = S.sprites.thumb, pr = S.probe; if (!img) return;
    const ts = pr.thumb.size, cols = pr.thumb.cols;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(img, (n % cols) * ts, Math.floor(n / cols) * ts, ts, ts, x, y, w, hh);
  };
  const CMAP = [[0, 0, 4], [50, 10, 94], [120, 28, 109], [187, 55, 85], [237, 105, 37], [252, 255, 164]];
  SE.cmap = function (v) {
    v = Math.max(0, Math.min(1, v));
    const x = v * (CMAP.length - 1), i = Math.min(CMAP.length - 2, Math.floor(x)), f = x - i, a = CMAP[i], b = CMAP[i + 1];
    return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
  };
  // mode: heat | tint | spot | glow
  SE.drawHeat = function (ctx, vals, g, w, hh, opts = {}) {
    if (!vals || !g) return;
    if (opts.mode === 'glow') {
      let lo = Infinity, hi = -Infinity; for (let p = 0; p < g * g; p++) { lo = Math.min(lo, vals[p]); hi = Math.max(hi, vals[p]); }
      if (!(hi > lo)) { lo = 0; hi = 1; }
      const N = 64, off = document.createElement('canvas'); off.width = off.height = N;
      const o = off.getContext('2d'), im = o.createImageData(N, N), a0 = opts.base ?? 0.25, a1 = opts.alpha ?? 0.5;
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        const fx = Math.max(0, Math.min(g - 1, (x + .5) / N * g - .5)), fy = Math.max(0, Math.min(g - 1, (y + .5) / N * g - .5));
        const x0 = Math.floor(fx), y0 = Math.floor(fy), x1 = Math.min(g - 1, x0 + 1), y1 = Math.min(g - 1, y0 + 1), ax = fx - x0, ay = fy - y0;
        const raw = (vals[y0 * g + x0] * (1 - ax) + vals[y0 * g + x1] * ax) * (1 - ay) + (vals[y1 * g + x0] * (1 - ax) + vals[y1 * g + x1] * ax) * ay;
        const v = (raw - lo) / (hi - lo), [r, gg, b] = SE.cmap(v);
        im.data.set([r, gg, b, Math.round((a0 + a1 * v) * 255)], (y * N + x) * 4);
      }
      o.putImageData(im, 0, 0);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(off, 0, 0, N, N, 0, 0, w, hh);
      return;
    }
    const off = document.createElement('canvas'); off.width = g; off.height = g;
    const o = off.getContext('2d'), im = o.createImageData(g, g), mode = opts.mode || 'heat', col = opts.color || [255, 80, 50];
    // stretch to this map's range
    let lo = 0, hi = 1;
    if (mode === 'spot') { lo = Infinity; hi = -Infinity; for (let p = 0; p < g * g; p++) { lo = Math.min(lo, vals[p]); hi = Math.max(hi, vals[p]); } if (!(hi > lo)) { lo = 0; hi = 1; } }
    for (let p = 0; p < g * g; p++) {
      let v = Math.max(0, Math.min(1, (vals[p] - lo) / (hi - lo)));
      if (mode === 'spot') v = v ** 1.5;
      let r, gg, b, a;
      if (mode === 'spot') { r = 0; gg = 0; b = 0; a = Math.round((1 - v) * (opts.alpha ?? 0.8) * 255); }
      else if (mode === 'tint') { [r, gg, b] = col; a = Math.round(v * v * (opts.alpha ?? 0.7) * 255); }
      else { [r, gg, b] = SE.cmap(v); a = Math.round((opts.alpha ?? 0.6) * 255); }
      im.data.set([r, gg, b, a], p * 4);
    }
    o.putImageData(im, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(off, 0, 0, g, g, 0, 0, w, hh);
  };
  // series: [{xs, ys, color, dash, label}]
  SE.chart = function (cv, series, opts = {}) {
    const [ctx, w, hh] = SE.fit(cv, opts.h || 160);
    ctx.clearRect(0, 0, w, hh);
    const padL = opts.axis ? 44 : 4, padR = opts.axis && series.length > 1 ? 44 : 6, padT = 8, padB = opts.axis ? 18 : 6;
    let x0 = Infinity, x1 = -Infinity;
    for (const s of series) for (let i = 0; i < s.xs.length; i++) if (s.ys[i] != null) { x0 = Math.min(x0, s.xs[i]); x1 = Math.max(x1, s.xs[i]); }
    if (!isFinite(x0)) return;
    if (opts.xmax != null) x1 = Math.max(x1, opts.xmax);
    const X = x => padL + (x - x0) / ((x1 - x0) || 1) * (w - padL - padR);
    const text = opts.text || SE.css('--muted') || '#888', grid = opts.grid || SE.css('--line') || '#333';
    ctx.font = '11px ' + (SE.css('--num-font') || 'system-ui');
    if (opts.axis) {
      ctx.strokeStyle = grid; ctx.lineWidth = 1;
      for (let i = 0; i <= 3; i++) { const y = padT + i * (hh - padT - padB) / 3; ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke(); }
      ctx.fillStyle = text; ctx.textAlign = 'left'; ctx.fillText('step ' + Math.round(x0).toLocaleString(), padL, hh - 4);
      ctx.textAlign = 'right'; ctx.fillText(Math.round(x1).toLocaleString(), w - padR, hh - 4);
    }
    series.forEach((s, si) => {
      const px = [], py = [];
      for (let i = 0; i < s.xs.length; i++) if (s.ys[i] != null && isFinite(s.ys[i])) { px.push(s.xs[i]); py.push(s.ys[i]); }
      if (!px.length) return;
      let y0 = Infinity, y1 = -Infinity; for (const y of py) { y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
      if (opts.yr) [y0, y1] = opts.yr;
      if (y1 - y0 < 1e-12) { y0 -= 1; y1 += 1; }
      const Y = y => padT + (y1 - y) / (y1 - y0) * (hh - padT - padB);
      const stride = Math.max(1, Math.floor(px.length / (w * 1.5)));
      ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6; ctx.setLineDash(s.dash || []); ctx.beginPath();
      for (let i = 0; i < px.length; i += stride) i ? ctx.lineTo(X(px[i]), Y(py[i])) : ctx.moveTo(X(px[i]), Y(py[i]));
      ctx.lineTo(X(px[px.length - 1]), Y(py[py.length - 1])); ctx.stroke(); ctx.setLineDash([]);
      if (opts.axis && si < 2 && !(opts.yr && si > 0)) {
        ctx.fillStyle = s.color; ctx.textAlign = si === 0 ? 'right' : 'left';
        const xx = si === 0 ? padL - 6 : w - padR + 6;
        ctx.fillText(SE.fmt(s.key, y1), xx, padT + 8); ctx.fillText(SE.fmt(s.key, y0), xx, hh - padB);
      }
      if (opts.xcur != null) {
        const v = SE.valueAt(px, py, opts.xcur);
        if (v != null) { ctx.fillStyle = s.color; ctx.beginPath(); ctx.arc(X(Math.min(opts.xcur, px[px.length - 1])), Y(v), 3, 0, 7); ctx.fill(); }
      }
    });
    for (const m of opts.markers || []) {
      ctx.strokeStyle = m.color; ctx.globalAlpha = 0.8; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(X(m.x), padT); ctx.lineTo(X(m.x), hh - padB); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    }
    if (opts.xcur != null) {
      ctx.strokeStyle = text; ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(X(opts.xcur), padT); ctx.lineTo(X(opts.xcur), hh - padB); ctx.stroke(); ctx.globalAlpha = 1;
    }
    cv._x = { x0, x1, padL, padR, w };
  };
  SE.chartStepAt = function (cv, clientX) {
    const r = cv.getBoundingClientRect(), m = cv._x; if (!m) return null;
    const f = (clientX - r.left - m.padL) / (m.w - m.padL - m.padR);
    return m.x0 + Math.max(0, Math.min(1, f)) * (m.x1 - m.x0);
  };
  SE.nearestSnap = function (step) {
    let best = -1, bd = Infinity;
    S.snaps.forEach((s, i) => { const d = Math.abs(s.step - step); if (d < bd) { bd = d; best = i; } });
    return best;
  };

  SE.delta = function (i, k) {
    const a = S.snaps[i - 1], b = S.snaps[i];
    if (!a || !b) return null;
    const x = (a.metrics || {})[k], y = (b.metrics || {})[k];
    return x == null || y == null ? null : y - x;
  };
  SE.verdict = function (i) {
    if (i == null) i = S.snaps.length - 1;
    const sn = S.snaps[i];
    if (!sn) return { level: 'wait', title: 'Waiting for the first snapshot', reasons: [] };
    const recent = S.alerts.filter(a => a.step <= sn.step && a.step > sn.step - 4 * ((S.meta && S.meta.snapshot_every) || 500));
    const bad = recent.filter(a => a.level === 'danger'), warn = recent.filter(a => a.level === 'warning');
    if (bad.length) return { level: 'bad', title: 'Training has a problem', reasons: bad.map(a => a.msg) };
    if (warn.length) return { level: 'warn', title: 'Worth watching', reasons: warn.map(a => a.msg) };
    const reasons = [];
    const dk = SE.delta(i, 'knn_acc'), dr = SE.delta(i, 'rankme'), ds = SE.delta(i, 'silhouette');
    const ns = (sn.metrics || {}).nbr_stability;
    if (dk != null) reasons.push(`kNN accuracy ${dk >= 0 ? 'up' : 'down'} ${Math.abs(dk * 100).toFixed(1)} points since the last snapshot`);
    else if (ds != null) reasons.push(`cluster quality ${ds >= 0 ? 'up' : 'down'} ${Math.abs(ds).toFixed(3)} since the last snapshot (no labels needed)`);
    if (dr != null) reasons.push(`effective rank ${dr >= 0 ? 'up' : 'down'} ${Math.abs(dr).toFixed(0)}`);
    if (dk == null && ns != null) reasons.push(`${(ns * 100).toFixed(0)}% of neighbours unchanged since the last snapshot`);
    return { level: 'ok', title: i === 0 ? 'First look at the untrained encoder' : 'Training looks healthy', reasons };
  };
  SE.story = function (i) {
    const sn = S.snaps[i]; if (!sn) return '';
    const m = sn.metrics || {};
    if (i === 0) return m.knn_acc != null
      ? `Before training. kNN accuracy ${SE.fmt('knn_acc', m.knn_acc)} (chance is ${S.probe && S.probe.class_names ? (100 / S.probe.class_names.length).toFixed(0) + '%' : 'low'}), effective rank ${SE.fmt('rankme', m.rankme)} of ${m.emb_dim}.`
      : `Before training. Effective rank ${SE.fmt('rankme', m.rankme)} of ${m.emb_dim}, cluster quality ${SE.fmt('silhouette', m.silhouette)}.`;
    const parts = [];
    const dk = SE.delta(i, 'knn_acc');
    if (dk != null) parts.push(Math.abs(dk) < 0.005 ? `kNN accuracy holds at ${SE.fmt('knn_acc', m.knn_acc)}` : `kNN accuracy ${dk > 0 ? 'rises' : 'falls'} ${Math.abs(dk * 100).toFixed(1)} points to ${SE.fmt('knn_acc', m.knn_acc)}`);
    const dr = SE.delta(i, 'rankme');
    if (dr != null && Math.abs(dr) >= 3) parts.push(`the embedding ${dr > 0 ? 'uses more' : 'loses'} dimensions (rank ${SE.fmt('rankme', m.rankme)})`);
    const ds = SE.delta(i, 'silhouette');
    if (ds != null && Math.abs(ds) >= 0.01) parts.push(`clusters ${ds > 0 ? 'sharpen' : 'blur'}`);
    let s = parts.join(', ');
    s = s ? (s.startsWith('kNN') ? s : s[0].toUpperCase() + s.slice(1)) + '.' : 'Little change.';
    return s;
  };

  window.SE = SE;
})();
