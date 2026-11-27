"""Input stats, token norms, per-layer attention and CKA."""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def input_stats(x, pre=None):
    x = x[:64].detach().float()
    out = {"in/raw_min": float(x.min()), "in/raw_max": float(x.max())}
    if pre is not None:
        x = pre(x)
    m, s = x.mean((0, 2, 3)), x.std((0, 2, 3))
    for i, c in enumerate("rgb"[: x.shape[1]]):
        out[f"in/mean_{c}"] = float(m[i])
        out[f"in/std_{c}"] = float(s[i])
    return out


@torch.no_grad()
def token_norms(tokens, ratio=2.5):
    """Outlier = norm > ratio x median."""
    n = tokens.float().norm(dim=-1)
    med = n.median().clamp_min(1e-6)
    frac = float((n > ratio * med).float().mean())
    rel = (n / n.amax(1, keepdim=True).clamp_min(1e-6)).cpu()
    return {"grid": int(math.isqrt(n.shape[1])), "ratio": ratio, "median": float(med), "max": float(n.max()),
            "data": [[round(float(v), 3) for v in row] for row in rel],
            "outlier": [[int(v) for v in row] for row in (n > ratio * med).cpu()]}, frac


def find_blocks(encoder):
    for m in encoder.modules():
        b = getattr(m, "blocks", None)
        if isinstance(b, (nn.Sequential, nn.ModuleList)) and len(b) > 0:
            return list(b)
    return []


VIT_SIZES = [("Tiny", 192, 12, 3), ("Small", 384, 12, 6), ("Base", 768, 12, 12), ("Large", 1024, 24, 16),
             ("Huge", 1280, 32, 16), ("giant", 1536, 40, 24), ("Gigantic", 1664, 48, 16)]


def model_info(encoder, total_params=None):
    """Param count, ViT shape and nearest standard size."""
    info = {"params_encoder": int(sum(p.numel() for p in encoder.parameters()))}
    if total_params is not None:
        info["params_total"] = int(total_params)
    blocks = find_blocks(encoder)
    if blocks:
        info["depth"] = len(blocks)
        dim = getattr(encoder, "embed_dim", None) or getattr(encoder, "num_features", None)
        if dim is None:
            for m in blocks[0].modules():
                if isinstance(m, nn.LayerNorm):
                    dim = m.normalized_shape[0]
                    break
        if dim:
            info["dim"] = int(dim)
        a = _attn_in(blocks[0])
        if a is not None and getattr(a, "num_heads", None):
            info["heads"] = int(a.num_heads)
        for m in encoder.modules():
            ps = getattr(m, "patch_size", None)
            if ps is not None:
                info["patch"] = int(ps[0] if isinstance(ps, (tuple, list)) else ps)
                break
        if "dim" in info:
            for name, d, L, _ in VIT_SIZES:
                if d == info["dim"]:
                    info["size"] = f"ViT-{name}"
                    if L != info["depth"]:
                        info["size_note"] = f"{info['depth']} of the usual {L} layers"
                    break
    return info


def _attn_in(block):
    found = None
    for m in block.modules():
        if hasattr(m, "num_heads") and not isinstance(m, nn.MultiheadAttention):
            name = type(m).__name__.lower()
            if "attn" in name or "attention" in name or isinstance(getattr(m, "qkv", None), nn.Linear):
                found = m
    return found


@torch.no_grad()
def layer_attention(encoder, x, n_prefix):
    """CLS attention per block, mean over heads: (B, L, P)."""
    blocks = find_blocks(encoder)
    mods = [_attn_in(b) for b in blocks]
    if not mods or any(m is None for m in mods):
        return None
    caps, hooks, fused, state = {}, [], [], {"cur": None}
    orig_sdpa = F.scaled_dot_product_attention
    for i, m in enumerate(mods):
        if hasattr(m, "fused_attn"):
            fused.append((m, m.fused_attn))
            m.fused_attn = False
        drop = getattr(m, "attn_drop", None)
        if isinstance(drop, nn.Module):
            hooks.append(drop.register_forward_hook(lambda mod, inp, out, i=i: caps.__setitem__(i, out.detach()) if out.dim() == 4 else None))
        hooks.append(m.register_forward_pre_hook(lambda mod, inp, i=i: state.__setitem__("cur", i)))
        hooks.append(m.register_forward_hook(lambda mod, inp, out: state.__setitem__("cur", None)))

    def sdpa(q, k, v, *a, **kw):
        i = state["cur"]
        if i is not None and i not in caps:
            sc = kw.get("scale") or q.shape[-1] ** -0.5
            kk = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1) if k.shape[1] != q.shape[1] else k
            caps[i] = (q.float() @ kk.float().transpose(-2, -1) * sc).softmax(-1)
        return orig_sdpa(q, k, v, *a, **kw)
    F.scaled_dot_product_attention = sdpa
    mea_mod = sys.modules.get(type(mods[0]).__module__)
    orig_mea = getattr(mea_mod, "memory_efficient_attention", None) if mea_mod is not None else None
    if callable(orig_mea):
        def mea(q, k, v, *a, **kw):
            i = state["cur"]
            if i is not None and i not in caps:
                qq, kk = q.transpose(1, 2).float(), k.transpose(1, 2).float()
                caps[i] = (qq @ kk.transpose(-2, -1) * (kw.get("scale") or qq.shape[-1] ** -0.5)).softmax(-1)
            return orig_mea(q, k, v, *a, **kw)
        setattr(mea_mod, "memory_efficient_attention", mea)
    try:
        encoder(x)
    finally:
        F.scaled_dot_product_attention = orig_sdpa
        if callable(orig_mea):
            setattr(mea_mod, "memory_efficient_attention", orig_mea)
        for h in hooks:
            h.remove()
        for m, f in fused:
            m.fused_attn = f
    if len(caps) != len(mods):
        return None
    out = []
    for i in range(len(mods)):
        a = caps[i].float()
        r = a[:, :, 0, n_prefix:] if n_prefix > 0 else a.mean(2)
        r = r.mean(1)
        out.append(r / r.sum(-1, keepdim=True).clamp_min(1e-12))
    res = torch.stack(out, 1)
    if int(math.isqrt(res.shape[-1])) ** 2 != res.shape[-1]:
        return None
    return res


@torch.no_grad()
def layer_features(encoder, x, n_prefix):
    """Mean-pooled output of each block."""
    blocks = find_blocks(encoder)
    if not blocks:
        return None
    caps, hooks = {}, []
    for i, b in enumerate(blocks):
        hooks.append(b.register_forward_hook(lambda m, inp, out, i=i: caps.__setitem__(i, out[0] if isinstance(out, (tuple, list)) else out)))
    try:
        encoder(x)
    finally:
        for h in hooks:
            h.remove()
    feats = []
    for i in range(len(blocks)):
        o = caps.get(i)
        if o is None:
            return None
        o = o.float()
        feats.append(o[:, n_prefix:].mean(1) if o.dim() == 3 else o.mean((2, 3)) if o.dim() == 4 else o.flatten(1))
    return feats


def linear_cka(x, y):
    x = x.double() - x.double().mean(0)
    y = y.double() - y.double().mean(0)
    hsic = (y.T @ x).pow(2).sum()
    den = (x.T @ x).pow(2).sum().sqrt() * (y.T @ y).pow(2).sum().sqrt()
    return float(hsic / den.clamp_min(1e-12))


def class_accuracy(pred, labels, n_classes):
    pred, labels = pred.cpu(), labels.cpu()
    return [round(float((pred[labels == c] == c).float().mean()), 4) if (labels == c).any() else None for c in range(n_classes)]
