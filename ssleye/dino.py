"""DINO head stats for collapse alerts."""
import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def head_stats(teacher_logits, student_logits=None, teacher_temp=0.04, student_temp=0.1, center=None):
    """Logits are (B, K). Returns entropy, top prototype share, prototypes used and KL."""
    t = teacher_logits.detach().float()
    if center is not None:
        t = t - center.detach().float()
    p = F.softmax(t / teacher_temp, dim=-1)
    k = p.shape[-1]
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean() / math.log(k)
    top = p.argmax(-1)
    counts = torch.bincount(top, minlength=k)
    out = {
        "dino/teacher_entropy": float(ent),
        "dino/top_proto_share": float(counts.max()) / max(1, top.numel()),
        "dino/protos_used": float((counts > 0).sum()) / k,
    }
    if student_logits is not None:
        logq = F.log_softmax(student_logits.detach().float() / student_temp, dim=-1)
        kl = (p * (torch.log(p.clamp_min(1e-12)) - logq)).sum(-1).mean()
        out["dino/kl"] = float(kl)
    return out
