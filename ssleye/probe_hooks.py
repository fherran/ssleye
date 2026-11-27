"""Features, patch tokens and CLS attention from any encoder.

Attention capture order: matrix (attn_drop), sdpa (q/k), qkv (fallback, ignores RoPE).
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


def _find_attn(encoder):
    found = None
    for m in encoder.modules():
        if not hasattr(m, "num_heads") or isinstance(m, nn.MultiheadAttention):
            continue
        name = type(m).__name__.lower()
        if "attn" in name or "attention" in name or isinstance(getattr(m, "qkv", None), nn.Linear):
            found = m
    return found


def _find_patch(encoder):
    for m in encoder.modules():
        blocks = getattr(m, "blocks", None)
        if isinstance(blocks, (nn.Sequential, nn.ModuleList)) and len(blocks) > 0:
            return blocks[-1]
    if hasattr(encoder, "layer4"):
        return encoder.layer4
    stages = getattr(encoder, "stages", None)
    if isinstance(stages, (nn.Sequential, nn.ModuleList)) and len(stages) > 0:
        return stages[-1]
    return None


def _patch_size(encoder):
    for m in encoder.modules():
        pe = getattr(m, "patch_embed", None)
        if pe is None:
            continue
        ps = getattr(pe, "patch_size", None)
        if ps is None:
            proj = getattr(pe, "proj", None)
            ps = getattr(proj, "kernel_size", None)
        if ps is not None:
            return tuple(ps) if isinstance(ps, (tuple, list)) else (int(ps), int(ps))
    ps = getattr(encoder, "patch_size", None)
    if ps is not None:
        return tuple(ps) if isinstance(ps, (tuple, list)) else (int(ps), int(ps))
    return None


class Inspector:
    def __init__(self, encoder, patch_layer=None, attn_module=None, methods=("matrix", "sdpa", "qkv")):
        self.methods = tuple(methods)
        self.encoder = encoder
        self.attn = attn_module if attn_module is not None else _find_attn(encoder)
        self.patch = patch_layer if patch_layer is not None else _find_patch(encoder)
        self.patch_size = _patch_size(encoder)
        self.n_prefix = None
        self.method = None
        self.warning = None

    def describe(self):
        return {
            "attention": type(self.attn).__name__ if self.attn is not None else None,
            "heads": int(self.attn.num_heads) if self.attn is not None else None,
            "patch_layer": type(self.patch).__name__ if self.patch is not None else None,
            "patch_size": list(self.patch_size) if self.patch_size else None,
            "prefix_tokens": self.n_prefix,
            "attention_method": self.method,
            "warning": self.warning,
        }

    @staticmethod
    def pooled(out):
        if isinstance(out, dict):
            for k in ("x_norm_clstoken", "pooler_output", "cls", "features"):
                if k in out:
                    out = out[k]
                    break
            else:
                out = next(iter(out.values()))
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() == 2:
            return out
        if out.dim() == 3:
            return out[:, 0]
        if out.dim() == 4:
            return out.mean((2, 3))
        return out.flatten(1)

    def features(self, x):
        return self.pooled(self.encoder(x)).float()

    def _prefix(self, n_tokens, x):
        if self.patch_size:
            gh, gw = x.shape[-2] // self.patch_size[0], x.shape[-1] // self.patch_size[1]
            pre = n_tokens - gh * gw
            if pre >= 0:
                return pre
        pre = getattr(self.encoder, "num_prefix_tokens", None)
        if pre is None:
            regs = getattr(self.encoder, "num_register_tokens", None) or getattr(self.encoder, "n_storage_tokens", 0) or 0
            pre = 1 + int(regs)
        return int(pre)

    def detail(self, x):
        cap = {}
        hooks = []
        attn = self.attn
        restore_fused = None
        mea_mod, orig_mea = None, None
        state = {"inside": False}
        orig_sdpa = F.scaled_dot_product_attention

        if attn is not None:
            drop = getattr(attn, "attn_drop", None)
            if "matrix" in self.methods and hasattr(attn, "fused_attn"):
                restore_fused = attn.fused_attn
                attn.fused_attn = False
            if "matrix" in self.methods and isinstance(drop, nn.Module):
                def drop_hook(m, i, o):
                    if isinstance(o, torch.Tensor) and o.dim() == 4:
                        cap["matrix"] = o.detach()
                hooks.append(drop.register_forward_hook(drop_hook))
            if "qkv" in self.methods and isinstance(getattr(attn, "qkv", None), nn.Linear):
                hooks.append(attn.qkv.register_forward_hook(lambda m, i, o: cap.__setitem__("qkv", o.detach())))
            hooks.append(attn.register_forward_pre_hook(lambda m, i: state.__setitem__("inside", True)))
            hooks.append(attn.register_forward_hook(lambda m, i, o: state.__setitem__("inside", False)))

            def sdpa(q, k, v, *args, **kwargs):
                if state["inside"] and "sdpa" in self.methods:
                    cap["sdpa"] = (q.detach(), k.detach(), kwargs.get("scale"))
                return orig_sdpa(q, k, v, *args, **kwargs)
            F.scaled_dot_product_attention = sdpa

            # xformers: patch it in the attention module's namespace, layout (B, N, H, D)
            mea_mod = sys.modules.get(type(attn).__module__)
            orig_mea = getattr(mea_mod, "memory_efficient_attention", None) if mea_mod is not None else None
            if callable(orig_mea):
                def mea(q, k, v, *args, **kwargs):
                    if state["inside"] and "sdpa" in self.methods:
                        cap["sdpa"] = (q.detach().transpose(1, 2), k.detach().transpose(1, 2), kwargs.get("scale"))
                    return orig_mea(q, k, v, *args, **kwargs)
                setattr(mea_mod, "memory_efficient_attention", mea)
            else:
                mea_mod = None

        if self.patch is not None:
            hooks.append(self.patch.register_forward_hook(
                lambda m, i, o: cap.__setitem__("patch", o[0] if isinstance(o, (tuple, list)) else o)))
        try:
            feats = self.features(x)
        finally:
            F.scaled_dot_product_attention = orig_sdpa
            if mea_mod is not None:
                setattr(mea_mod, "memory_efficient_attention", orig_mea)
            for h in hooks:
                h.remove()
            if restore_fused is not None:
                attn.fused_attn = restore_fused

        a = None
        if "matrix" in cap:
            a, self.method = cap["matrix"].float(), "matrix"
        elif "sdpa" in cap:
            q, k, scale = cap["sdpa"]
            scale = scale if scale is not None else q.shape[-1] ** -0.5
            if k.shape[1] != q.shape[1]:  # grouped-query attention
                k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            a, self.method = (q.float() @ k.float().transpose(-2, -1) * scale).softmax(-1), "sdpa"
        elif "qkv" in cap:
            o = cap["qkv"]
            b, n, c3 = o.shape
            h = int(attn.num_heads)
            qkv = o.reshape(b, n, 3, h, c3 // 3 // h).permute(2, 0, 3, 1, 4)
            qq, kk = qkv[0], qkv[1]
            qn, kn = getattr(attn, "q_norm", None), getattr(attn, "k_norm", None)
            if isinstance(qn, nn.Module):
                qq = qn(qq)
            if isinstance(kn, nn.Module):
                kk = kn(kk)
            scale = float(getattr(attn, "scale", qq.shape[-1] ** -0.5))
            a, self.method = (qq.float() @ kk.float().transpose(-2, -1) * scale).softmax(-1), "qkv"

        n_tokens = a.shape[-1] if a is not None else (cap["patch"].shape[1] if isinstance(cap.get("patch"), torch.Tensor) and cap["patch"].dim() == 3 else None)
        if n_tokens is not None:
            self.n_prefix = self._prefix(n_tokens, x)

        attn_map = None
        if a is not None:
            npf = self.n_prefix or 0
            attn_map = a[:, :, 0, npf:] if npf > 0 else a.mean(2)
            attn_map = attn_map / attn_map.sum(-1, keepdim=True).clamp_min(1e-12)
            if int(math.isqrt(attn_map.shape[-1])) ** 2 != attn_map.shape[-1]:
                attn_map = None

        tokens = None
        p = cap.get("patch")
        if isinstance(p, torch.Tensor):
            if p.dim() == 4:
                tokens = p.flatten(2).transpose(1, 2)
            elif p.dim() == 3:
                tokens = p[:, (self.n_prefix or 0):]
        if tokens is not None and int(math.isqrt(tokens.shape[1])) ** 2 != tokens.shape[1]:
            tokens = None
        return feats, tokens, attn_map
