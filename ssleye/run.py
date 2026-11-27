"""Run logger, writes to ssleye_runs/<project>/<run>/."""
import atexit
import shutil
import base64
import json
import math
import os
import statistics
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import metrics as M
from . import profiles as P
from . import extras as X
from .downstream import Downstream
from .probe_hooks import Inspector

DEFAULT_ALERTS = {
    "loss_spike": 2.5,        # x median loss
    "grad_spike": 10.0,       # x median grad norm
    "rank_drop": 0.30,        # fraction below best
    "knn_drop": 0.05,
    "std_collapse": 0.15,
    "plateau_rel": 0.005,
    "plateau_window": 1000,
    "cooldown": 300,          # steps between repeats
    "dino_uniform": 0.95,
    "dino_one_proto": 0.5,
    "dense_drop": 0.30,
}


def _f(v):
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        v = v.detach().float().item()
    v = float(v)
    return v if math.isfinite(v) else None


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=str))


def _replace(tmp, path):
    for i in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (i + 1))
    os.replace(tmp, path)


def _atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    _replace(tmp, path)


def _sprite(imgs, cols, size=None):
    x = imgs.detach().float().cpu().clamp(0, 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = x[:, :3]
    if size and (x.shape[-1] != size or x.shape[-2] != size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False,
                          antialias=size < x.shape[-1])
    n, _, h, w = x.shape
    rows = max(1, math.ceil(n / cols))
    canvas = torch.zeros(3, rows * h, cols * w)
    for i in range(n):
        r, c = divmod(i, cols)
        canvas[:, r * h:(r + 1) * h, c * w:(c + 1) * w] = x[i]
    arr = (canvas.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def _round_robin(labels, n):
    groups = [(labels == c).nonzero().flatten().tolist() for c in sorted(set(labels.tolist()))]
    out = []
    longest = max(len(g) for g in groups)
    for i in range(longest):
        for g in groups:
            if i < len(g):
                out.append(g[i])
    return out[:n]


def _b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


class Run:
    def __init__(self, project="default", name=None, dir="ssleye_runs", config=None,
                 snapshot_every=500, stream_every=25, stream_n=6, alerts=None, model="auto",
                 stages=None, batch_info=None, sample_every_min=30, sample_n=24, distributed=False):
        self.model, self.profile = P.get(model)
        self.rank, self.world = 0, 1
        if distributed:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self._dist = self.world > 1
        # other ranks make the same calls (FSDP needs every rank in the forward) but write to a temp dir
        self._follower = self.rank != 0
        name = name or time.strftime("run-%Y%m%d-%H%M%S")
        if self._follower:
            import tempfile
            path = Path(tempfile.mkdtemp(prefix=f"ssleye-rank{self.rank}-"))
        else:
            base = Path(dir) / project
            path = base / name
            if path.exists():
                path = base / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
        (path / "snapshots").mkdir(parents=True)
        (path / "stream").mkdir()
        self.path = path
        self.project = project
        self.name = path.name
        self.snapshot_every = int(snapshot_every)
        self.stream_every = int(stream_every)
        self.stream_n = int(stream_n)
        self.alert_cfg = {**DEFAULT_ALERTS, **(alerts or {})}
        self.meta = {
            "project": project, "name": self.name, "created": time.time(), "status": "running",
            "config": _jsonable(config or {}), "snapshot_every": self.snapshot_every,
            "stream_every": self.stream_every, "alerts": self.alert_cfg, "config_keys": [],
            "inspector": None, "version": "0.3.0", "model": self.model, "profile": self.profile, "warnings": [],
            "stages": _jsonable(stages or [{"name": "Pre-training", "note": "single stage"}]), "stage": 0,
            "batch_info": _jsonable(batch_info or {}), "pipeline": None,
            "sample_every_min": float(sample_every_min), "sample_n": int(sample_n),
            "world_size": self.world,
        }
        self.sample_n = int(sample_n)
        self._sample_every = float(sample_every_min) * 60.0
        self._next_sample = 0.0
        self._sample_seen = {}
        self._write_meta()
        self._files = {k: open(path / f"{k}.jsonl", "a", encoding="utf-8")
                       for k in ("scalars", "snapshots", "alerts", "stream", "samples", "downstream")}
        self._insp = None
        self._probe = None
        self._pre = lambda x: x
        self._aug = None
        self._bs = 256
        self._loss_hist = deque(maxlen=200)
        self._gn_hist = deque(maxlen=200)
        self._loss_steps = deque()
        self._last_alert = {}
        self._best = {}
        self._prev_emb = None
        self._prev_margin = None
        self._last_t = None
        self._last_step = None
        self._ips = None
        self._last_logged = None
        self._last_snap = None
        self._disp_ver = 0
        self._t_ready = None
        self._t_prev_end = None
        self._pre_given = False
        self._down = []
        self._layer_hist = []
        self._prev_topi = None
        self._gram_ref = None
        self._settings_mtime = None
        self._probe_meta = None
        self._dsize = None
        self._finished = False
        atexit.register(self._atexit)
        self._say(f"[ssleye] logging to {self.path.resolve()}", flush=True)

    def _say(self, msg, **kw):
        if not self._follower:
            print(msg, flush=True)

    def _agree(self, flag):
        """Rank 0 decides, so every rank enters the same forward passes."""
        if not self._dist:
            return flag
        import torch.distributed as dist
        dev = torch.device("cuda", torch.cuda.current_device()) if dist.get_backend() == "nccl" else "cpu"
        t = torch.tensor([int(flag)], device=dev)
        dist.broadcast(t, 0)
        return bool(t.item())

    def _write_meta(self):
        _atomic_json(self.path / "meta.json", self.meta)

    def _append(self, kind, obj):
        f = self._files[kind]
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        f.flush()

    def _alert(self, step, level, kind, msg):
        last = self._last_alert.get(kind)
        if last is not None and step - last < self.alert_cfg["cooldown"]:
            return
        self._last_alert[kind] = step
        self._append("alerts", {"step": int(step), "t": time.time(), "level": level,
                                "type": kind, "msg": msg})
        self._say(f"[ssleye] {level.upper()} step {step}: {msg}", flush=True)

    def set_probe(self, images, labels=None, class_names=None, n_display=16, n_matrix=50, n_views=128):
        """Fixed images checked at every snapshot. labels are optional."""
        x = images.detach().float().cpu()
        if x.max() > 1.5:
            x = x / 255.0
        n = len(x)
        lab = None if labels is None else torch.as_tensor(labels).long().cpu()
        if lab is not None:
            order = _round_robin(lab, n)
        else:
            order = torch.randperm(n, generator=torch.Generator().manual_seed(0)).tolist()
        disp = order[:n_display]
        matrix = order[:n_matrix]
        if lab is not None:
            matrix = sorted(matrix, key=lambda i: (int(lab[i]), i))
        views_idx = order[:n_views]
        h = x.shape[-1]
        thumb = min(48, h)
        cols = 40
        _sprite(x, cols, thumb).save(self.path / "probe_sprite.png")
        dsize = min(128, max(h, 32))
        _sprite(x[disp], len(disp), dsize).save(self.path / "display_sprite.png")
        if lab is not None and class_names is None:
            class_names = [str(i) for i in range(int(lab.max()) + 1)]
        self._probe = {"x": x, "labels": lab, "disp": disp, "matrix": matrix, "views": views_idx, "order": order,
                       "n_classes": int(lab.max()) + 1 if lab is not None else None}
        self._dsize = dsize
        self._probe_meta = {
            "n": n, "labels": lab.tolist() if lab is not None else None,
            "class_names": list(class_names) if class_names else None,
            "display_idx": disp, "matrix_idx": matrix,
            "thumb": {"file": "probe_sprite.png", "cols": cols, "size": thumb},
            "display": {"file": "display_sprite.png", "size": dsize},
            "display_versions": [{"version": 0, "file": "display_sprite.png", "idx": disp, "from_step": 0}],
        }
        _atomic_json(self.path / "probe.json", self._probe_meta)

    def _apply_settings(self, step):
        """Reload settings.json saved by the dashboard."""
        p = self.path / "settings.json"
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            return
        if mtime == self._settings_mtime:
            return
        self._settings_mtime = mtime
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            self._say(f"[ssleye] could not read settings.json: {e}", flush=True)
            return
        probe = self._probe
        n = len(probe["x"])
        if cfg.get("mode") == "manual":
            idx = []
            for i in cfg.get("display_idx", []):
                i = int(i)
                if 0 <= i < n and i not in idx:
                    idx.append(i)
            idx = idx[:64]
        else:
            k = max(1, min(64, int(cfg.get("n_display", len(probe["disp"])))))
            idx = probe["order"][:k]
        if not idx or idx == probe["disp"]:
            return
        self._disp_ver += 1
        fname = f"display_sprite_v{self._disp_ver}.png"
        _sprite(probe["x"][idx], len(idx), self._dsize).save(self.path / fname)
        probe["disp"] = idx
        self._prev_margin = None
        pm = self._probe_meta
        pm["display_idx"] = idx
        pm["display"] = {"file": fname, "size": self._dsize}
        pm["display_versions"].append({"version": self._disp_ver, "file": fname, "idx": idx, "from_step": int(step)})
        _atomic_json(self.path / "probe.json", pm)
        self._say(f"[ssleye] settings applied at step {step}: {len(idx)} images for attention/similarity maps", flush=True)

    def watch(self, encoder, preprocess=None, augment=None, patch_layer=None, attn_module=None,
              batch_size=256, total_params=None):
        """augment is used for LiDAR and align/uniform. total_params = everything being trained."""
        if isinstance(encoder, torch.nn.parallel.DistributedDataParallel):
            encoder = encoder.module
        self._insp = Inspector(encoder, patch_layer, attn_module)
        try:
            self.meta["model_info"] = X.model_info(encoder, total_params)
        except Exception as e:
            self.meta["model_info"] = {"error": str(e)[:200]}
        self._pre = preprocess or (lambda x: x)
        self._aug = augment
        self._bs = int(batch_size)
        self._pre_given = preprocess is not None
        self.meta["inspector"] = self._insp.describe()
        self._write_meta()

    def log(self, step, loss=None, config=None, metrics=None, batch=None, epoch=None, samples=None):
        """Call every step. batch: dict or list of images in [0, 1]."""
        step = int(step)
        now = time.time()
        rec = {"step": step, "t": now}
        if epoch is not None:
            rec["epoch"] = _f(epoch)
        raw_loss = None
        if loss is not None:
            raw_loss = float(loss.detach().float().item()) if isinstance(loss, torch.Tensor) else float(loss)
            rec["loss"] = raw_loss if math.isfinite(raw_loss) else None
        for k, v in (config or {}).items():
            rec["cfg/" + k] = _f(v)
            if k not in self.meta["config_keys"]:
                self.meta["config_keys"].append(k)
                self._write_meta()
        for k, v in (metrics or {}).items():
            rec[k] = _f(v)
        if samples and self._last_t is not None and now > self._last_t and step > self._last_step:
            ips = samples * (step - self._last_step) / (now - self._last_t)
            self._ips = ips if self._ips is None else 0.9 * self._ips + 0.1 * ips
            rec["img_per_s"] = self._ips
        self._last_t, self._last_step = now, step
        if self._t_ready is not None and self._t_prev_end is not None and self._t_ready >= self._t_prev_end:
            rec["time/data_ms"] = round((self._t_ready - self._t_prev_end) * 1000, 2)
            rec["time/compute_ms"] = round((now - self._t_ready) * 1000, 2)
        self._t_ready = None
        if batch is not None and self.stream_every > 0 and step % self.stream_every == 0:
            st = self._safe("input_stats", self._input_stats, step, batch)
            if st:
                rec.update(st)
        self._append("scalars", rec)
        self._last_logged = step

        self._train_alerts(step, raw_loss, rec.get("grad_norm"), (metrics or {}).get("grad_norm"))
        self._dino_alerts(step, rec)

        if batch is not None and self.stream_every > 0 and step % self.stream_every == 0:
            self._stream(step, batch)
        if self._insp is not None and self._probe is not None and step % self.snapshot_every == 0:
            self.snapshot(step)
        if self._insp is not None:
            for d in self._down:
                if step % d.every == 0 and d.last_step != step:
                    self._run_downstream(step, d)
        self._t_prev_end = time.time()

    def mark_data_ready(self):
        """Call after loading the batch to split data vs compute time."""
        self._t_ready = time.time()

    def _input_stats(self, step, batch):
        t = next(iter(batch.values())) if isinstance(batch, dict) else (batch[0] if isinstance(batch, (list, tuple)) else batch)
        if t.dim() == 5:
            t = t[:, 0]
        st = X.input_stats(t, self._pre if self._pre_given else None)
        if st["in/raw_max"] > 1.5:
            self._alert(step, "danger", "input_range", f"Inputs go up to {st['in/raw_max']:.0f}: they look like 0-255 pixels, not 0-1. Divide by 255.")
        if self._pre_given:
            means = [v for k, v in st.items() if k.startswith("in/mean_")]
            stds = [v for k, v in st.items() if k.startswith("in/std_")]
            if any(abs(m) > 1.0 for m in means) or any(s < 0.2 or s > 5 for s in stds):
                self._alert(step, "warning", "input_norm", "After normalisation the inputs are far from mean 0 / std 1: check the preprocessing.")
        return st

    def add_downstream(self, name, task="classification", train=None, test=None, every=None, class_names=None, few_shot=0.1):
        """train/test: (images, labels)."""
        d = Downstream(name, task, train, test, every or self.snapshot_every * 5, class_names=class_names, few_shot=few_shot)
        self._down.append(d)
        self.meta.setdefault("downstream", []).append(d.describe())
        self._write_meta()
        return d

    def _run_downstream(self, step, d):
        res = self._safe("downstream", d.evaluate, self._insp, self._pre)
        d.last_step = step
        if res:
            res.update(step=int(step), t=time.time())
            self._append("downstream", res)
            self._say(f"[ssleye] downstream {d.name} @ {step}: kNN {res['knn']:.3f} linear {res['linear']:.3f} few-shot {res['fewshot']:.3f} ({res['eval_s']}s)", flush=True)

    def _cka(self, step, feats):
        cur = [f.detach().float().cpu() for f in feats]
        n = len(cur)
        ll = [[round(X.linear_cka(cur[i], cur[j]), 4) for j in range(n)] for i in range(n)]
        self._layer_hist.append((int(step), [c.half() for c in cur]))
        lt = [[round(X.linear_cka(h[l].float(), cur[l]), 4) for (_, h) in self._layer_hist] for l in range(n)]
        return {"layers": n, "ll": ll, "steps": [s for s, _ in self._layer_hist], "lt": lt}

    def _train_alerts(self, step, loss, gn, gn_raw):
        cfg = self.alert_cfg
        if loss is not None:
            if not math.isfinite(loss):
                self._alert(step, "danger", "loss_nan", f"Loss is {loss}")
            else:
                if len(self._loss_hist) >= 50:
                    med = statistics.median(self._loss_hist)
                    if med > 0 and loss > cfg["loss_spike"] * med:
                        self._alert(step, "danger", "loss_spike",
                                    f"Loss spiked {loss / med:.1f}x the recent median ({loss:.3f} vs {med:.3f})")
                self._loss_hist.append(loss)
                self._loss_steps.append((step, loss))
                w = cfg["plateau_window"]
                while self._loss_steps and self._loss_steps[0][0] < step - 2 * w:
                    self._loss_steps.popleft()
        if gn_raw is not None:
            g = _f(gn_raw)
            if g is None:
                self._alert(step, "danger", "grad_nan", "Gradient norm is NaN/inf")
            else:
                if len(self._gn_hist) >= 50:
                    med = statistics.median(self._gn_hist)
                    if med > 0 and g > cfg["grad_spike"] * med:
                        self._alert(step, "danger", "grad_spike",
                                    f"Gradient norm {g:.1f} is {g / med:.0f}x its recent median ({med:.2f})")
                self._gn_hist.append(g)

    def _dino_alerts(self, step, rec):
        # only alert after the head was healthy once
        cfg = self.alert_cfg
        ent = rec.get("dino/teacher_entropy")
        if ent is not None and ent < cfg["dino_uniform"] - 0.15:
            self._best["dino_ent_ok"] = True
        if ent is not None and ent > cfg["dino_uniform"] and self._best.get("dino_ent_ok"):
            self._alert(step, "danger", "dino_uniform",
                        f"DINO uniform collapse: teacher output is almost flat (entropy {ent:.2f} of max 1). Lower the teacher temperature or check centring.")
        share = rec.get("dino/top_proto_share")
        if share is not None and share < cfg["dino_one_proto"] / 2:
            self._best["dino_share_ok"] = True
        if share is not None and share > cfg["dino_one_proto"] and self._best.get("dino_share_ok"):
            self._alert(step, "danger", "dino_one_proto",
                        f"DINO one-prototype collapse: {share:.0%} of the batch picks the same prototype. Check centring momentum and teacher temperature.")

    def _plateau_check(self, step):
        w = self.alert_cfg["plateau_window"]
        if step < 3 * w or not self._loss_steps:
            return
        old = [l for s, l in self._loss_steps if step - 2 * w <= s < step - w]
        new = [l for s, l in self._loss_steps if s >= step - w]
        if len(old) < 20 or len(new) < 20:
            return
        a, b = sum(old) / len(old), sum(new) / len(new)
        if a > 0 and (a - b) / a < self.alert_cfg["plateau_rel"]:
            self._alert(step, "warning", "plateau",
                        f"Loss barely moved over the last {2 * w} steps ({a:.4f} to {b:.4f})")

    def _stream(self, step, batch):
        if self._follower:
            return
        if isinstance(batch, dict):
            names, tensors = list(batch.keys()), list(batch.values())
        else:
            tensors = list(batch)
            names = [f"view {i + 1}" for i in range(len(tensors))]
        n = min(self.stream_n, min(t.shape[0] for t in tensors))
        size = min(64, max(32, max(t.shape[-1] for t in tensors)))
        rows = []
        for t in tensors:  # resize multi-crop views to one size
            t = t[:n].detach().float().cpu()
            if t.shape[1] == 1:
                t = t.repeat(1, 3, 1, 1)
            if t.shape[-2:] != (size, size):
                t = F.interpolate(t[:, :3], size=(size, size), mode="bilinear", align_corners=False)
            rows.append(t[:, :3])
        rows = torch.cat(rows, 0)
        fname = f"stream/{step:08d}.png"
        _sprite(rows, n, size).save(self.path / fname)
        self._append("stream", {"step": step, "file": fname, "rows": names, "n": n, "size": size})

    @torch.no_grad()
    def snapshot(self, step):
        t0 = time.time()
        self._apply_settings(step)
        insp, probe = self._insp, self._probe
        enc = insp.encoder
        was_training = enc.training
        enc.eval()
        try:
            dev = next(enc.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        x_all = probe["x"]
        extra_layer_attn = extra_layer_feats = None
        try:
            feats = []
            for i in range(0, len(x_all), self._bs):
                feats.append(insp.features(self._pre(x_all[i:i + self._bs].to(dev))))
            z = torch.cat(feats)
            _, tokens, attn = insp.detail(self._pre(x_all[probe["disp"]].to(dev)))
            npf = insp.n_prefix or 0
            extra_layer_attn = self._safe("layer_attention", X.layer_attention, enc, self._pre(x_all[probe["disp"]].to(dev)), npf)
            extra_layer_feats = self._safe("layer_features", X.layer_features, enc, self._pre(x_all[probe["order"][:256]].to(dev)), npf)
            views = None
            if self._aug is not None:
                sub = x_all[probe["views"]].to(dev)
                views = torch.stack([insp.features(self._pre(self._aug(sub))) for _ in range(8)], 1)
        finally:
            enc.train(was_training)

        if not torch.isfinite(z).all():
            bad = int((~torch.isfinite(z)).any(1).sum())
            self._alert(step, "danger", "nan_features",
                        f"The encoder outputs NaN/inf for {bad} of {len(z)} probe images, so this snapshot was skipped. Training has diverged.")
            return

        info = insp.describe()
        if info != self.meta.get("inspector"):
            self.meta["inspector"] = info
            self.meta["warnings"] = P.check(self.profile, info)
            for w in self.meta["warnings"]:
                self._say(f"[ssleye] WARNING: {w}", flush=True)
            self._write_meta()

        labels = probe["labels"]
        met = {"emb_dim": int(z.shape[1])}
        safe = self._safe
        met["rankme"] = safe("rankme", M.rankme, z)
        ev = safe("spectrum", M.eigen_spectrum, z)
        if ev is None:
            ev = torch.zeros(1, dtype=torch.float64)
        met["alpha"] = safe("alpha", M.alpha_req, ev)
        met["std_ratio"] = safe("std_ratio", M.std_ratio, z)
        if views is not None:
            met["lidar"] = safe("lidar", M.lidar, views)
            au = safe("align_uniform", M.align_uniform, views[:, 0], views[:, 1])
            met["align"], met["uniform"] = au if au is not None else (None, None)
        kn = M.knn(z, labels)
        if "acc" in kn:
            met["knn_acc"] = kn["acc"]
        # neighbours kept since the last snapshot
        topi = kn["nbrs"]
        stable = None
        if self._prev_topi is not None and self._prev_topi.shape == topi.shape:
            prev = self._prev_topi.to(topi.device)
            stable = (topi[:, :, None] == prev[:, None, :]).any(-1).float().mean(1)
            met["nbr_stability"] = float(stable.mean())
        self._prev_topi = topi.cpu()
        zn = F.normalize(z, dim=1)
        zn_np = zn.cpu().numpy().astype(np.float64)
        cq = safe("silhouette", M.cluster_quality, zn_np, probe["n_classes"] or 10)
        sil, clusters = cq if cq is not None else (None, None)
        met["silhouette"] = sil
        if tokens is not None:
            safe("patch_metrics", self._dense_metrics, step, tokens, met)
        extra = {}
        if tokens is not None:
            tn = safe("token_norms", X.token_norms, tokens)
            if tn is not None:
                extra["token_norm"], met["outlier_tokens"] = tn
        if extra_layer_attn is not None:
            la = extra_layer_attn / extra_layer_attn.amax(-1, keepdim=True).clamp_min(1e-12)
            extra["layer_attn"] = {"layers": int(la.shape[1]), "grid": int(math.isqrt(la.shape[-1])),
                                   "data": _b64((la.cpu().numpy() * 255 + 0.5).astype(np.uint8))}
        if extra_layer_feats is not None:
            cka = safe("cka", self._cka, step, extra_layer_feats)
            if cka is not None:
                extra["cka"] = cka
        if labels is not None and "pred" in kn:
            extra["class_acc"] = X.class_accuracy(kn["pred"], labels, probe["n_classes"])
        met = {k: (_f(v) if not isinstance(v, int) else v) for k, v in met.items()}

        emb = M.pca2(zn_np)
        if self._prev_emb is not None:
            emb = M.procrustes(emb, self._prev_emb)
        self._prev_emb = emb

        mi = probe["matrix"]
        simmat = (zn[mi] @ zn[mi].T).cpu().numpy()
        spec = ev[:64]
        spec = (spec / (ev.sum() + 1e-12)).cpu().numpy()

        disp = probe["disp"]
        margin = kn.get("margin")
        disp_margin = [float(margin[i]) for i in disp] if margin is not None else None
        display = []
        for j, i in enumerate(disp):
            d = {"idx": int(i), "nbrs": [int(v) for v in kn["nbrs"][i, :5].tolist()]}
            if stable is not None:
                d["stable"] = round(float(stable[i]), 3)
            if clusters is not None:
                d["cluster"] = int(clusters[i])
            if margin is not None:
                d["pred"] = int(kn["pred"][i])
                d["correct"] = bool(kn["pred"][i] == labels[i].to(kn["pred"].device))
                d["margin"] = round(float(margin[i]), 3)
                if self._prev_margin is not None:
                    d["delta"] = round(float(margin[i]) - self._prev_margin[j], 3)
            display.append(d)
        self._prev_margin = disp_margin

        snap = {"step": int(step), "t": time.time(), "metrics": met, "display_version": self._disp_ver,
                "emb2d": np.round(emb, 3).tolist(),
                "emb_color": labels.tolist() if labels is not None else (
                    clusters.tolist() if clusters is not None else None),
                "simmat": {"n": len(mi), "data": np.round(simmat, 3).flatten().tolist()},
                "spectrum": [float(f"{v:.6g}") for v in spec.tolist()],
                "display": display}
        if attn is not None:
            a = attn.float()
            a = a / a.amax(-1, keepdim=True).clamp_min(1e-12)
            snap["attn"] = {"heads": int(a.shape[1]), "grid": int(math.isqrt(a.shape[-1])),
                            "data": _b64((a.cpu().numpy() * 255 + 0.5).astype(np.uint8))}
        if tokens is not None:
            t = F.normalize(tokens.float(), dim=-1)
            snap["tokens"] = {"dim": int(t.shape[-1]), "grid": int(math.isqrt(t.shape[1])),
                              "data": _b64(np.clip(np.round(t.cpu().numpy() * 127), -127, 127).astype(np.int8))}
        snap.update(extra)
        met["snapshot_ms"] = round((time.time() - t0) * 1000)
        fname = f"snapshots/{step:08d}.json"
        _atomic_json(self.path / fname, snap)
        disp_stable = [round(float(stable[i]), 3) for i in disp] if stable is not None else None
        self._append("snapshots", {"step": int(step), "t": snap["t"], "file": fname, "display_version": self._disp_ver,
                                   "metrics": met, "disp_margin": disp_margin, "disp_stable": disp_stable,
                                   "class_acc": extra.get("class_acc")})
        self._last_snap = int(step)
        self._snapshot_alerts(step, met)
        self._plateau_check(step)

    def _safe(self, name, fn, *args):
        try:
            return fn(*args)
        except Exception as e:  # noqa: BLE001
            fails = self.meta.setdefault("metric_failures", {})
            fails[name] = fails.get(name, 0) + 1
            if fails[name] == 1:
                self._say(f"[ssleye] WARNING: metric '{name}' failed and was skipped ({type(e).__name__}: {str(e)[:160]})", flush=True)
            self._write_meta()
            return None

    def _dense_metrics(self, step, tokens, met):
        """patch_locality = neighbour cosine - mean cosine; gram_drift vs the best snapshot."""
        t = F.normalize(tokens.float(), dim=-1)
        sim = t @ t.transpose(1, 2)
        p = sim.shape[-1]
        g = int(math.isqrt(p))
        idx = torch.arange(p, device=sim.device)
        r, c = idx // g, idx % g
        adj = ((r[:, None] - r[None]).abs() + (c[:, None] - c[None]).abs()) == 1
        off = ~torch.eye(p, dtype=torch.bool, device=sim.device)
        loc = float(sim[:, adj].mean() - sim[:, off].mean())
        met["patch_locality"] = loc
        ref = self._gram_ref
        if ref is not None and ref["ver"] == self._disp_ver and ref["G"].shape == sim.shape:
            G = ref["G"].to(sim.device)
            met["gram_drift"] = float(((sim - G).flatten(1).norm(dim=1) / G.flatten(1).norm(dim=1).clamp_min(1e-12)).mean())
            if self.profile.get("dense_watch") and ref["loc"] > 0.02 and loc < (1 - self.alert_cfg["dense_drop"]) * ref["loc"]:
                self._alert(step, "warning", "dense_degrade",
                            f"Patch features are degrading: neighbouring patches are {1 - loc / ref['loc']:.0%} less distinct than at step {ref['step']}. "
                            "This is the failure DINOv3's Gram anchoring prevents.")
        # skip the first snapshots, step 0 can look sharp
        warmed = getattr(self, "_snap_count", 0) >= 3
        if warmed and (ref is None or ref["ver"] != self._disp_ver or loc > ref["loc"]):
            self._gram_ref = {"ver": self._disp_ver, "G": sim.detach().cpu(), "loc": loc, "step": int(step)}

    def _snapshot_alerts(self, step, met):
        cfg = self.alert_cfg
        self._snap_count = getattr(self, "_snap_count", 0) + 1
        r = met.get("rankme")
        # rank dips early on, wait a few snapshots
        if r is not None and self._snap_count > 3:
            best = self._best.get("rankme", r)
            self._best["rankme"] = max(best, r)
            if r < (1 - cfg["rank_drop"]) * self._best["rankme"]:
                self._alert(step, "warning", "rank_drop",
                            f"Effective rank fell to {r:.0f} from its best {self._best['rankme']:.0f}: possible dimensional collapse")
        s = met.get("std_ratio")
        if s is not None:
            best = max(self._best.get("std_ratio", s), s)
            self._best["std_ratio"] = best
            # only after it was healthy
            if s < cfg["std_collapse"] and best >= 2 * cfg["std_collapse"]:
                self._alert(step, "danger", "collapse",
                            f"Embedding spread fell to {s:.3f} from {best:.3f} (healthy is near 1): the encoder is collapsing")
        k = met.get("knn_acc")
        if k is not None:
            best = max(self._best.get("knn_acc", k), k)
            self._best["knn_acc"] = best
            if best - k > cfg["knn_drop"]:
                self._alert(step, "warning", "knn_drop",
                            f"kNN accuracy dropped to {k:.1%} from its best {best:.1%}")

    def set_pipeline(self, networks):
        """[{"name": "Student", "views": ["global 1", ...]}, ...]; optional "note", "stage_from"."""
        self.meta["pipeline"] = _jsonable(networks)
        self._write_meta()

    def set_stage(self, index, step=None):
        """Move to stage `index` of init(stages=...)."""
        index = int(index)
        if index != self.meta.get("stage"):
            self.meta["stage"] = index
            self.meta.setdefault("stage_starts", {})[str(index)] = int(step) if step is not None else self._last_logged
            self._write_meta()
            name = (self.meta["stages"][index]["name"] if index < len(self.meta["stages"]) else index)
            self._say(f"[ssleye] stage {index + 1}: {name}", flush=True)

    def sample_due(self):
        """True every sample_every_min minutes."""
        return self._agree(time.time() >= self._next_sample)

    def log_samples(self, step, samples):
        from .samples import save_samples
        self._next_sample = time.time() + self._sample_every
        try:
            save_samples(self, step, samples)
        except Exception as e:  # noqa: BLE001
            self._say(f"[ssleye] WARNING: batch samples skipped ({type(e).__name__}: {str(e)[:160]})", flush=True)

    def finish(self, step=None, snapshot=True):
        if self._finished:
            return
        last = step if step is not None else self._last_logged
        if snapshot and last is not None and self._insp is not None and self._probe is not None \
                and last != self._last_snap:
            self.snapshot(last)
        if snapshot and last is not None and self._insp is not None:
            for d in self._down:
                if d.last_step != last:
                    self._run_downstream(last, d)
        self._finished = True
        self.meta["status"] = "finished"
        self.meta["finished"] = time.time()
        self._write_meta()
        for f in self._files.values():
            f.close()
        if self._follower:
            shutil.rmtree(self.path, ignore_errors=True)

    def _atexit(self):
        if not self._finished:
            try:
                self.finish(snapshot=False)
            except Exception:
                pass


def init(project="default", name=None, **kw):
    return Run(project=project, name=name, **kw)
