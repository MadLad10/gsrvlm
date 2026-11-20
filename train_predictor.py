"""E6 step 2: predict log(teacher s_med) from cached vision features.

Self-supervised: the target is the teacher's scale, never benchmark GT.
Train scenes only; 10% of them held out for early stopping / model choice.
With ~170 training scenes, a ridge probe on mean-pooled features is the
honest main model; the small frame-transformer from the plan is reported too.

    python train_predictor.py          # CPU is fine; uses GPU if present
Writes cache/learned_scale.json (eval scenes) and results/predictor.json.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from scalehint import ROOT_DEFAULTS, data


class FrameTransformer(nn.Module):
    def __init__(self, d_in, d=256, layers=2, heads=4):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))

    def forward(self, x):
        return self.head(self.enc(self.inp(x)).mean(1)).squeeze(-1)


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return lambda Xn: ((Xn - mu) / sd) @ w + ym


def train_tf(Xtr, ytr, Xva, yva, seed, dev, epochs=200, patience=20):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = FrameTransformer(Xtr.shape[-1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.05)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    Xv, yv = torch.tensor(Xva, device=dev), torch.tensor(yva, device=dev)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), device=dev)
        for i in range(0, len(Xt), 32):
            b = perm[i:i + 32]
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = (m(Xt[b]) - yt[b]).abs().mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        m.eval()
        with torch.no_grad():
            v = (m(Xv) - yv).abs().mean().item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    return m, best, ep + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="cache/qwen_vis")
    ap.add_argument("--teacher", default=ROOT_DEFAULTS["teacher"])
    ap.add_argument("--split_file", default=ROOT_DEFAULTS["split"])
    ap.add_argument("--out", default=ROOT_DEFAULTS["learned"])
    ap.add_argument("--report", default=os.path.join(ROOT_DEFAULTS["results"], "predictor.json"))
    a = ap.parse_args()

    sp = data.load_split(a.split_file)
    with open(a.teacher) as f:
        T = {s: v["s_med"] for s, v in json.load(f).items() if v.get("s_med")}
    assert not set(sp["train"]) & set(sp["eval"]), "split leaks scenes"

    def load(scenes):
        S, X, y = [], [], []
        for s in scenes:
            p = os.path.join(a.feats, f"{s}.pt")
            if s in T and os.path.exists(p):
                S.append(s)
                X.append(torch.load(p).float().numpy())
                y.append(np.log(T[s]))
        return S, np.stack(X).astype(np.float32), np.array(y, np.float32)

    Str, Xtr_all, ytr_all = load(sp["train"])
    Sev, Xev, yev = load(sp["eval"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Str))
    nva = max(5, int(round(0.1 * len(Str))))
    va, tr = perm[:nva], perm[nva:]
    print(f"train {len(tr)} / val {len(va)} / eval {len(Sev)} scenes, feat {Xtr_all.shape[1:]}")

    rep = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_eval": len(Sev),
           "feat_shape": list(Xtr_all.shape[1:]), "target": "log teacher s_med"}

    def err(p, y):
        return float(np.median(np.abs(p - y)))

    # constant
    c = float(ytr_all[tr].mean())
    rep["constant"] = {"val": err(np.full(len(va), c), ytr_all[va]),
                       "eval": err(np.full(len(Sev), c), yev)}

    # ridge on mean-pooled frames, alpha picked on val
    P = Xtr_all.mean(1)
    best = None
    for alpha in [1, 10, 100, 1e3, 1e4, 1e5]:
        f = ridge_fit(P[tr], ytr_all[tr], alpha)
        v = err(f(P[va]), ytr_all[va])
        if best is None or v < best[0]:
            best = (v, alpha)
    t0 = time.time()
    fr = ridge_fit(P[np.concatenate([tr, va])], ytr_all[np.concatenate([tr, va])], best[1])
    ridge_eval = fr(Xev.mean(1))
    rep["ridge"] = {"alpha": best[1], "val": best[0], "eval": err(ridge_eval, yev),
                    "params": int(P.shape[1] + 1), "train_s": round(time.time() - t0, 3)}

    # frame transformer, 3 seeds, ensemble
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    preds_ev, preds_va, vals, eps = [], [], [], []
    for seed in (0, 1, 2):
        m, v, ep = train_tf(Xtr_all[tr], ytr_all[tr], Xtr_all[va], ytr_all[va], seed, dev)
        with torch.no_grad():
            preds_ev.append(m(torch.tensor(Xev, device=dev)).float().cpu().numpy())
            preds_va.append(m(torch.tensor(Xtr_all[va], device=dev)).float().cpu().numpy())
        vals.append(v)
        eps.append(ep)
    tf_eval = np.mean(preds_ev, 0)
    rep["transformer"] = {
        "params": int(sum(p.numel() for p in m.parameters())),
        "val": err(np.mean(preds_va, 0), ytr_all[va]), "val_per_seed": vals,
        "eval": err(tf_eval, yev), "epochs": eps,
        "train_s_total_3_seeds": round(time.time() - t0, 1), "device": dev,
        "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if dev == "cuda" else None,
    }

    # choose on validation only
    pick = "ridge" if rep["ridge"]["val"] <= rep["transformer"]["val"] else "transformer"
    final = ridge_eval if pick == "ridge" else tf_eval
    rep["chosen"] = pick
    rep["note"] = "errors are median |log(pred) - log(teacher s_med)|"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({s: float(np.exp(p)) for s, p in zip(Sev, final)}, f, indent=1)
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
