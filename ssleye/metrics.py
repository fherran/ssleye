"""Representation metrics."""
import math

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-7


def _entropy_rank(values):
    values = values.clamp_min(0)
    p = values / (values.sum() + EPS) + EPS
    return float(torch.exp(-(p * torch.log(p)).sum()))


def rankme(z):
    """RankMe (Garrido et al. 2023)."""
    return _entropy_rank(torch.linalg.svdvals(z.double()))


def eigen_spectrum(z):
    z = z.detach().double().cpu()
    zc = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(zc)
    return (s * s / max(1, zc.shape[0] - 1)).clamp_min(0)


def alpha_req(ev):
    """alpha-ReQ (Agrawal et al. 2022), fit on ranks 2..n/2."""
    if ev.numel() == 0 or float(ev[0]) <= 0:
        return None
    ev = ev[ev > ev[0] * 1e-7]
    n = ev.numel()
    if n < 6:
        return None
    lo, hi = 1, max(4, n // 2)
    k = torch.arange(1, n + 1, dtype=torch.float64, device=ev.device)
    x = torch.log(k[lo:hi])
    y = torch.log(ev[lo:hi])
    xc = x - x.mean()
    slope = (xc * (y - y.mean())).sum() / (xc * xc).sum()
    return float(-slope)


def std_ratio(z):
    """~1 when spread out, ~0 when collapsed."""
    zn = F.normalize(z.float(), dim=1)
    return float(zn.std(0).mean() * math.sqrt(zn.shape[1]))


def align_uniform(z1, z2):
    """Wang & Isola 2020."""
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)
    align = (z1 - z2).pow(2).sum(1).mean()
    sq = torch.pdist(z1, p=2).pow(2)
    uniform = sq.mul(-2).exp().mean().log()
    return float(align), float(uniform)


def lidar(views, delta=1e-4, eps=1e-8):
    """LiDAR (Thilak et al. 2023). views: (n, q, D)."""
    n, q, d = views.shape
    v = views.detach().double().cpu()
    mu_i = v.mean(1)
    mu = mu_i.mean(0)
    db = mu_i - mu
    sb = db.T @ db / max(1, n - 1)
    dw = (v - mu_i[:, None]).reshape(-1, d)
    sw = dw.T @ dw / max(1, n * (q - 1))
    sw = sw + delta * torch.eye(d, dtype=v.dtype, device=v.device)
    evw, u = torch.linalg.eigh(sw)
    inv_sqrt = u @ torch.diag(evw.clamp_min(eps).rsqrt()) @ u.T
    m = inv_sqrt @ sb @ inv_sqrt
    m = (m + m.T) / 2
    return _entropy_rank(torch.linalg.eigvalsh(m))


def knn(z, labels=None, k=20, temperature=0.07):
    """Leave-one-out cosine kNN."""
    zn = F.normalize(z.float(), dim=1)
    sim = zn @ zn.T
    sim.fill_diagonal_(-2.0)
    k = min(k, z.shape[0] - 1)
    topv, topi = sim.topk(k, dim=1)
    out = {"nbrs": topi}
    if labels is not None:
        labels = labels.to(z.device)
        n_classes = int(labels.max()) + 1
        nl = labels[topi]
        votes = torch.zeros(z.shape[0], n_classes, device=z.device)
        votes.scatter_add_(1, nl, torch.exp(topv / temperature))
        pred = votes.argmax(1)
        out["pred"] = pred
        out["acc"] = float((pred == labels).float().mean())
        out["margin"] = (nl == labels[:, None]).float().mean(1)
    return out


def cluster_quality(zn_np, n_clusters):
    """k-means + cosine silhouette."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception:
        return None, None
    km = KMeans(n_clusters=n_clusters, n_init=3, random_state=0).fit(zn_np)
    if len(set(km.labels_.tolist())) < 2:
        return 0.0, km.labels_
    return float(silhouette_score(zn_np, km.labels_, metric="cosine")), km.labels_


def pca2(zn_np):
    zc = zn_np - zn_np.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    y = zc @ vt[:2].T
    return y / (y.std() + EPS)


def procrustes(y, ref):
    """Align y to ref so the map doesn't jump."""
    u, _, vt = np.linalg.svd(y.T @ ref)
    return y @ (u @ vt)
