# utils_debug.py
import torch, math
import torch.nn.functional as F

def tstats(name, x, max_items=3):
    x = x.detach()
    mean = x.mean().item()
    std  = x.std().item()
    l2   = x.norm(dim=-1).mean().item() if x.ndim>1 else x.norm().item()
    return f"{name}: shape={tuple(x.shape)} mean={mean:.4f} std={std:.4f} l2(avg)={l2:.4f}"

def entropy(p, dim=-1, eps=1e-12):
    p = torch.clamp(p, eps, 1.0)
    return -(p * p.log()).sum(dim=dim)

def topk_neg_sim(z1, z2, k=5):
    z1n = F.normalize(z1, dim=-1); z2n = F.normalize(z2, dim=-1)
    S = z1n @ z2n.t()
    B = S.size(0)
    S = S - torch.eye(B, device=S.device)*1e9
    vals, _ = S.topk(k, dim=-1)
    return vals.mean().item()

def hist_counts(x, bins=(-0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)):
    x = x.detach().flatten().clamp(-1,1)
    counts = []
    for i in range(len(bins)-1):
        l, r = bins[i], bins[i+1]
        counts.append(int(((x>=l)&(x<r)).sum().item()))
    return dict(zip([f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)], counts))

class StepGate:
    def __init__(self, every=800):
        self.every = every
        self.cnt = 0
    def __call__(self):
        self.cnt += 1
        return (self.cnt % self.every)==0
