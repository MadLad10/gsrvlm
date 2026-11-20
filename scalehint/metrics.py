import numpy as np
import pandas as pd

# VSI-Bench MRA: thresholds 0.50, 0.55, ..., 0.95; correct at theta when
# |pred - gt| / gt <= 1 - theta (the official code uses <=).
THETAS = np.linspace(0.5, 0.95, 10)


def mra(pred, gt):
    if pred is None or gt is None or not np.isfinite(pred) or gt <= 0:
        return 0.0
    rel = abs(pred - gt) / gt
    return float(np.mean(rel <= 1.0 - THETAS + 1e-12))


def mra_vec(pred, gt):
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    ok = np.isfinite(pred) & (gt > 0)
    rel = np.where(ok, np.abs(pred - gt) / np.where(gt > 0, gt, 1), np.inf)
    return (rel[:, None] <= 1.0 - THETAS[None, :] + 1e-12).mean(1)


def scene_bootstrap(df, stat, n=1000, seed=0, scene_col="scene_id"):
    """95% CI of stat(df) resampling whole scenes with replacement."""
    rng = np.random.default_rng(seed)
    groups = {s: g for s, g in df.groupby(scene_col)}
    keys = list(groups)
    vals = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys), replace=True)
        # a scene drawn twice becomes two scenes, so per-scene stats stay valid
        b = pd.concat([groups[keys[i]].assign(**{scene_col: f"{keys[i]}#{j}"})
                       for j, i in enumerate(pick)], ignore_index=True)
        try:
            v = stat(b)
        except Exception:
            continue
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
