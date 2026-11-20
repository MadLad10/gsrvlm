"""E5: does the VLM use the scale cue?

For a hint source (teacher or anchor) with conditions <src>_k0.5, _k0.71, <src>
(k=1), _k1.41, _k2: per question, beta = slope of log(pred) on log(k).
beta = 1: the answer scales fully with the hint; beta = 0: hint ignored.

Copy rate: share of answers within 5% of the number written in the hint
(converted m <-> cm when the hint and answer units differ). Room size answers
(m2) are compared to the written number as is.

    python elasticity.py [--sources teacher,anchor]
Writes results/elasticity.json and figures/elasticity_<src>.pdf/png.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from scalehint import ROOT_DEFAULTS, TASKS, results
from scalehint.metrics import scene_bootstrap
from scalehint.prompts import K_LEVELS


def cond_name(src, k):
    return src if k == 1.0 else f"{src}_k{k:g}"


def copy_rate(df, src):
    d = df[df["ok"] & df["hint_value"].notna()]
    if d.empty:
        return np.nan
    hint_unit = "m" if src in ("teacher", "learned", "shuffled") else "cm"

    def to_hint_unit(row):
        u = TASKS[row["task"]]
        if u == hint_unit or u == "m2":
            return row["pred"]
        return row["pred"] * 100 if (u, hint_unit) == ("m", "cm") else row["pred"] / 100

    p = d.apply(to_hint_unit, axis=1)
    return float((np.abs(p - d["hint_value"]) / d["hint_value"] <= 0.05).mean())


def per_question_beta(frames):
    """frames: {k: df}. Returns df with qid, scene_id, task, beta, n_levels."""
    rows = []
    allk = pd.concat([f.assign(k=k) for k, f in frames.items()], ignore_index=True)
    allk = allk[allk["ok"]]
    for (qid, sid, task), g in allk.groupby(["qid", "scene_id", "task"]):
        if g["k"].nunique() < 3:
            continue
        beta = np.polyfit(np.log(g["k"].values), np.log(g["pred"].values), 1)[0]
        rows.append({"qid": qid, "scene_id": sid, "task": task, "beta": beta,
                     "n_levels": int(g["k"].nunique())})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="teacher,anchor")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(ROOT_DEFAULTS["results"], "elasticity.json"))
    a = ap.parse_args()

    os.makedirs("figures", exist_ok=True)
    rep = {}
    for src in [s.strip() for s in a.sources.split(",") if s.strip()]:
        ks = sorted(set(K_LEVELS) | {1.0})
        frames = {k: results.load(cond_name(src, k)) for k in ks}
        frames = {k: f for k, f in frames.items() if not f.empty}
        if len(frames) < 3:
            print(f"{src}: only {len(frames)} k-levels on disk, skipping")
            continue
        B = per_question_beta(frames)
        R = {"k_levels": sorted(frames), "tasks": {}}
        for t, g in B.groupby("task"):
            R["tasks"][t] = {
                "n": int(len(g)),
                "median_beta": float(g["beta"].median()),
                "median_beta_ci": scene_bootstrap(g, lambda x: x["beta"].median(), a.n_boot),
                "frac_ignore(beta<0.2)": float((g["beta"] < 0.2).mean()),
                "frac_full(beta>0.8)": float((g["beta"] > 0.8).mean()),
            }
        R["copy_rate"] = {}
        for k, f in frames.items():
            R["copy_rate"][f"{k:g}"] = {t: copy_rate(g, src) for t, g in f.groupby("task")}
        rep[src] = R
        print(src, json.dumps(R, indent=1, default=float))

        # Figure: median log(pred/gt) vs log k per task, with slope-1 reference
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        allk = pd.concat([f.assign(k=k) for k, f in frames.items()], ignore_index=True)
        allk = allk[allk["ok"]]
        common = allk.groupby("qid")["k"].nunique()
        allk = allk[allk["qid"].isin(common[common == len(frames)].index)]
        for t, g in allk.groupby("task"):
            m = g.groupby("k")["e"].median()
            x = np.log2(m.index.values)
            ax.plot(x, m.values, "o-", label=t.replace("object_", "").replace("_estimation", ""), ms=3)
            y1 = m.get(1.0, np.nan)
            ax.plot(x, y1 + x * np.log(2), ":", color=ax.lines[-1].get_color(), lw=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("log2 k (hint multiplier)")
        ax.set_ylabel("median log(pred/gt)")
        ax.set_title(f"{src} hint; dotted = full use (beta=1)", fontsize=8)
        ax.legend(fontsize=6)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"figures/elasticity_{src}.{ext}", dpi=200)
        plt.close(fig)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, default=float)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
