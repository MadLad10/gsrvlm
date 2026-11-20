"""E2: how much of the baseline log-error is shared within a scene?

e = log(pred / gt) per parseable question. Reported per task (and absolute
distance split at 2 m):

  slope_b       OLS slope of e on log(gt). b = 0: answers scale with the truth;
                b = -1: the model answers a constant (readout collapse).
  icc_raw       one-way ANOVA ICC(1) of e by scene.
  icc_resid     same, after removing a + b*log(gt). Readout collapse makes
                big-room scenes look "scaled down" together, which inflates
                icc_raw; icc_resid is the conservative number.
  mra_*         on scenes with >= 2 parseable questions of the task:
                base          baseline MRA (unparseable = 0)
                global        pred / exp(mean e of OTHER scenes)   (task-level bias only)
                scene_loo     pred / exp(mean e of other questions in the SAME scene)
                scene_loo_res same with residuals: removes only scene-shared error
                scene_loo - global is the part a per-scene scale could add on
                top of a constant correction.

    python decompose.py [--cond baseline] [--n_boot 1000]
Writes results/decomposition.json and prints GO / NO-GO for absolute distance.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from scalehint import ROOT_DEFAULTS, results
from scalehint.metrics import mra_vec, scene_bootstrap


def icc1(df, col):
    d = df[df[col].notna()]
    d = d.groupby("scene_id").filter(lambda g: len(g) >= 2)
    if d["scene_id"].nunique() < 3:
        return np.nan
    g = d.groupby("scene_id")[col]
    n_i, m_i = g.size().values, g.mean().values
    N, k = n_i.sum(), len(n_i)
    grand = d[col].mean()
    ssb = (n_i * (m_i - grand) ** 2).sum()
    ssw = ((d[col] - g.transform("mean")) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    sb = (msb - msw) / n0
    return float(sb / (sb + msw)) if (sb + msw) > 0 else np.nan


def naive_share(df, col):
    d = df[df[col].notna()].groupby("scene_id").filter(lambda g: len(g) >= 2)
    if len(d) < 3:
        return np.nan
    sm = d.groupby("scene_id")[col].transform("mean")
    return float(sm.var() / d[col].var())


def add_resid(df):
    d = df.copy()
    ok = d["e"].notna()
    x = np.log(d.loc[ok, "gt"].values)
    y = d.loc[ok, "e"].values
    b, a = np.polyfit(x, y, 1) if ok.sum() >= 3 else (np.nan, np.nan)
    d["r"] = np.nan
    d.loc[ok, "r"] = y - (a + b * x)
    return d, float(b)


def loo_mras(df):
    """MRA after the three rescalings, on scenes with >= 2 parseable questions."""
    d = df.copy()
    okc = d.groupby("scene_id")["ok"].transform("sum")
    d = d[okc >= 2].copy()
    if d.empty:
        return {}
    out = {"n": int(len(d)), "n_scenes": int(d["scene_id"].nunique())}
    out["base"] = float(mra_vec(d["pred"].fillna(np.nan), d["gt"]).mean())

    for col, name in (("e", "scene_loo"), ("r", "scene_loo_res")):
        s = d.groupby("scene_id")[col].transform("sum")
        n = d.groupby("scene_id")[col].transform("count")
        c = (s - d[col].fillna(0)) / (n - d[col].notna().astype(int))
        p = d["pred"] / np.exp(c)
        out[name] = float(mra_vec(p.where(d["ok"]), d["gt"]).mean())

    # leave-scene-out global mean of e
    tot, cnt = d["e"].sum(), d["e"].count()
    ss = d.groupby("scene_id")["e"].transform("sum")
    sc = d.groupby("scene_id")["e"].transform("count")
    cg = (tot - ss) / (cnt - sc)
    out["global"] = float(mra_vec((d["pred"] / np.exp(cg)).where(d["ok"]), d["gt"]).mean())
    return out


def analyse(df, n_boot):
    d, b = add_resid(df)
    res = {
        "n": int(len(d)), "n_scenes": int(d["scene_id"].nunique()),
        "unparseable_rate": float(1 - d["ok"].mean()),
        "mra_all": float(d["mra"].mean()),
        "slope_b": b,
        "share_naive_raw": naive_share(d, "e"),
        "icc_raw": icc1(d, "e"),
        "icc_resid": icc1(d, "r"),
        "median_abs_log_err": float(d["e"].abs().median()),
    }
    res["mra"] = loo_mras(d)
    if n_boot:
        res["icc_raw_ci"] = scene_bootstrap(d, lambda x: icc1(x, "e"), n_boot)
        res["icc_resid_ci"] = scene_bootstrap(d, lambda x: icc1(add_resid(x)[0], "r"), n_boot)

        def gain(x):
            m = loo_mras(x)
            return m["scene_loo_res"] - m["global"] if m else np.nan
        res["gain_scene_over_global_ci"] = scene_bootstrap(d, gain, n_boot)
    return res


def cross_task(df):
    """Do scenes that are over-estimated on one task get over-estimated on others?
    Spearman across scenes of per-task scene-mean residuals."""
    parts = {}
    for t, g in df.groupby("task"):
        g, _ = add_resid(g)
        parts[t] = g.groupby("scene_id")["r"].mean()
    out = {}
    ts = sorted(parts)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = parts[ts[i]], parts[ts[j]]
            common = a.index.intersection(b.index)
            if len(common) >= 10:
                rho = pd.Series(a[common]).rank().corr(pd.Series(b[common]).rank())
                out[f"{ts[i]}~{ts[j]}"] = {"spearman": float(rho), "n_scenes": int(len(common))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="baseline")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(ROOT_DEFAULTS["results"], "decomposition.json"))
    a = ap.parse_args()

    df = results.load(a.cond)
    if df.empty:
        raise SystemExit(f"no results for {a.cond}")
    rep = {"condition": a.cond, "tasks": {}}
    for t, g in df.groupby("task"):
        rep["tasks"][t] = analyse(g, a.n_boot)
        print(t, json.dumps({k: v for k, v in rep["tasks"][t].items() if not k.endswith("_ci")}, default=float))
    dist = df[df["task"] == "object_abs_distance"]
    if len(dist):
        rep["tasks"]["object_abs_distance<2m"] = analyse(dist[dist["gt"] < 2], a.n_boot)
        rep["tasks"]["object_abs_distance>=2m"] = analyse(dist[dist["gt"] >= 2], a.n_boot)
    rep["cross_task"] = cross_task(df)
    print("cross-task:", rep["cross_task"])

    if "object_abs_distance" in rep["tasks"]:
        r = rep["tasks"]["object_abs_distance"]
        m = r["mra"]
        loo_gain = m.get("scene_loo", np.nan) - m.get("base", np.nan)
        extra = m.get("scene_loo_res", np.nan) - m.get("global", np.nan)
        go = (r["icc_raw"] >= 0.4) or (loo_gain >= 0.05)
        rep["go_no_go"] = {"icc_raw": r["icc_raw"], "icc_resid": r["icc_resid"],
                           "scene_loo_gain_over_base": loo_gain,
                           "scene_resid_gain_over_global": extra,
                           "decision": "GO" if go else "NO-GO"}
        print("\nabs distance:", rep["go_no_go"])
        if go and (r["icc_resid"] < 0.2 or extra < 0.02):
            print("!! GO on raw numbers, but most of it is readout collapse / global bias, "
                  "not scene-specific scale. Report to Wasif before E4.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
