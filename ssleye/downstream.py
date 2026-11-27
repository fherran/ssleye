"""Frozen-encoder probes (kNN, linear, few-shot) at checkpoints."""
import time

import torch
import torch.nn.functional as F

TASKS = ("classification",)


class Downstream:
    def __init__(self, name, task, train, test, every, class_names=None, few_shot=0.1, seed=0):
        if task not in TASKS:
            raise ValueError(f"task {task!r} is not supported yet; supported: {', '.join(TASKS)}. "
                             "Segmentation and detection probes need a labelled dense dataset (planned).")
        self.name, self.task, self.every, self.few_shot = name, task, int(every), float(few_shot)
        self.xtr, self.ytr = self._prep(train)
        self.xte, self.yte = self._prep(test)
        self.class_names = list(class_names) if class_names else None
        self.n_classes = int(max(self.ytr.max(), self.yte.max())) + 1
        g = torch.Generator().manual_seed(seed)
        n = len(self.ytr)
        self.few_idx = torch.randperm(n, generator=g)[:max(self.n_classes, int(n * self.few_shot))]
        self.last_step = None

    @staticmethod
    def _prep(pair):
        x, y = pair
        x = x.detach().cpu()
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        return x.float(), torch.as_tensor(y).long().cpu()

    def describe(self):
        return {"name": self.name, "task": self.task, "every": self.every, "n_train": len(self.ytr), "n_test": len(self.yte),
                "few_shot": self.few_shot, "class_names": self.class_names}

    @torch.no_grad()
    def _feats(self, insp, pre, x, bs, dev):
        out = []
        for i in range(0, len(x), bs):
            out.append(insp.features(pre(x[i:i + bs].to(dev))).float())
        return torch.cat(out)

    def _linear(self, ftr, ytr, fte, yte, epochs=300):
        mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True).clamp_min(1e-6)
        a, b = (ftr - mu) / sd, (fte - mu) / sd
        with torch.enable_grad():
            lin = torch.nn.Linear(a.shape[1], self.n_classes).to(a.device)
            opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-4)
            for _ in range(epochs):
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(lin(a), ytr).backward()
                opt.step()
        with torch.no_grad():
            pred = lin(b).argmax(1)
        return float((pred == yte).float().mean()), pred

    def evaluate(self, insp, pre, bs=512):
        t0 = time.time()
        enc = insp.encoder
        was = enc.training
        enc.eval()
        try:
            dev = next(enc.parameters()).device
            ftr = self._feats(insp, pre, self.xtr, bs, dev)
            fte = self._feats(insp, pre, self.xte, bs, dev)
        finally:
            enc.train(was)
        ytr, yte = self.ytr.to(ftr.device), self.yte.to(ftr.device)
        a, b = F.normalize(ftr, dim=1), F.normalize(fte, dim=1)
        sim = b @ a.T
        topv, topi = sim.topk(20, dim=1)
        votes = torch.zeros(len(yte), self.n_classes, device=ftr.device).scatter_add_(1, ytr[topi], torch.exp(topv / 0.07))
        knn = float((votes.argmax(1) == yte).float().mean())
        lin, pred = self._linear(ftr, ytr, fte, yte)
        fi = self.few_idx.to(ftr.device)
        few, _ = self._linear(ftr[fi], ytr[fi], fte, yte)
        per_class = [round(float((pred[yte == c] == c).float().mean()), 4) if (yte == c).any() else None for c in range(self.n_classes)]
        return {"name": self.name, "task": self.task, "knn": round(knn, 4), "linear": round(lin, 4),
                "fewshot": round(few, 4), "per_class": per_class, "eval_s": round(time.time() - t0, 2)}
