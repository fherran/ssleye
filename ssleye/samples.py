"""Batch samples for the training view.

Sample format:
    {"id": int, "image": (3,H,W) or (T,3,H,W), "kind": "image"|"clip",
     "views": [{"name", "kind", "image", "box": (x,y,w,h) or None, "mask": hidden-patch grid or None}],
     "outputs": {"agree", "recon", "error", "pred"}}   # all optional
"""
import base64
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _png(t, path, size=None):
    x = t.detach().float().cpu().clamp(0, 1)
    if x.dim() == 4:  # clip -> strip of frames
        x = torch.cat(list(x), dim=2)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    if size and x.shape[-2] != size:
        h, w = x.shape[-2:]
        x = F.interpolate(x[None, :3], size=(size, int(round(w * size / h))), mode="bilinear", align_corners=False)[0]
    arr = (x[:3].permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _f(v):
    v = float(v)
    return round(v, 4) if math.isfinite(v) else None


def save_samples(run, step, samples):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rel = f"samples/{int(step):08d}"
    d = run.path / rel
    d.mkdir(parents=True, exist_ok=True)
    seen_before = run._sample_seen
    out = []
    originals = []
    for k, s in enumerate(samples):
        sid = int(s["id"])
        img = s["image"]
        kind = s.get("kind", "clip" if img.dim() == 4 else "image")
        _png(img, d / f"{k}_orig.png", size=96)
        rec = {"id": sid, "kind": kind, "frames": int(img.shape[0]) if img.dim() == 4 else 1,
               "seen": sid in seen_before, "seen_at": seen_before.get(sid), "views": [], "outputs": {}}
        originals.append(img[img.shape[0] // 2] if img.dim() == 4 else img)
        for j, v in enumerate(s.get("views", [])):
            vi = v["image"]
            fn = f"{k}_v{j}.png"
            _png(vi, d / fn, size=64)
            m = v.get("mask")
            rec["views"].append({
                "name": v["name"], "kind": v.get("kind", "view"), "file": f"{rel}/{fn}",
                "px": int(vi.shape[-1]), "frames": int(vi.shape[0]) if vi.dim() == 4 else 1,
                "box": [_f(t) for t in v["box"]] if v.get("box") is not None else None,
                "mask": torch.as_tensor(m).to(torch.uint8).cpu().tolist() if m is not None else None,
            })
        o = s.get("outputs") or {}
        if "agree" in o:
            a = o["agree"]
            rec["outputs"]["agree"] = {"a": [_f(t) for t in a["a"]], "b": [_f(t) for t in a["b"]],
                                       "a_label": a.get("a_label", "A"), "b_label": a.get("b_label", "B"),
                                       "value": _f(a["value"]), "metric": a.get("metric", "match")}
        if "recon" in o:
            _png(o["recon"], d / f"{k}_recon.png", size=96)
            rec["outputs"]["recon"] = f"{rel}/{k}_recon.png"
        if "error" in o:
            rec["outputs"]["error"] = [[_f(t) for t in row] for row in torch.as_tensor(o["error"]).tolist()]
        if "pred" in o:
            rec["outputs"]["pred"] = [{"label": p["label"], "value": _f(p["value"])} for p in o["pred"]]
        rec["orig"] = f"{rel}/{k}_orig.png"
        out.append(rec)
        seen_before[sid] = time.strftime("%H:%M")

    maps = {}
    if run._insp is not None and originals:
        try:
            dev = next(run._insp.encoder.parameters()).device
            x = torch.stack([o.float() for o in originals]).to(dev)
            enc = run._insp.encoder
            was = enc.training
            enc.eval()
            with torch.no_grad():
                _, tokens, attn = run._insp.detail(run._pre(x))
            enc.train(was)
            if attn is not None:
                a = attn.float()
                a = a / a.amax(-1, keepdim=True).clamp_min(1e-12)
                maps["attn"] = {"heads": int(a.shape[1]), "grid": int(math.isqrt(a.shape[-1])),
                                "data": _b64((a.cpu().numpy() * 255 + 0.5).astype(np.uint8))}
            if tokens is not None:
                t = F.normalize(tokens.float(), dim=-1)
                maps["tokens"] = {"dim": int(t.shape[-1]), "grid": int(math.isqrt(t.shape[1])),
                                  "data": _b64(np.clip(np.round(t.cpu().numpy() * 127), -127, 127).astype(np.int8))}
        except Exception as e:  # noqa: BLE001
            run._say(f"[ssleye] WARNING: sample attention maps skipped ({type(e).__name__}: {str(e)[:120]})", flush=True)

    index = {"step": int(step), "t": time.time(), "clock": time.strftime("%H:%M"), "stage": run.meta.get("stage"),
             "samples": out, "maps": maps}
    with open(d / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, separators=(",", ":"))
    run._append("samples", {"step": int(step), "t": index["t"], "clock": index["clock"], "file": f"{rel}/index.json",
                            "n": len(out), "stage": run.meta.get("stage")})
    run._say(f"[ssleye] saved {len(out)} batch samples at step {step} ({stamp})", flush=True)
