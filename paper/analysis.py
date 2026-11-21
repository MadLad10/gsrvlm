"""Every number and figure in the paper that is not already in results/tables.md.

    python paper/analysis.py [--n_boot 1000]          (cwd = repo root)

Reads results/ (Qwen3-VL-4B) and results_8b/ (Qwen3-VL-8B, if present).
Writes paper/numbers.json and paper/fig_compression.pdf, paper/fig_elasticity.pdf.
All CIs are 95% percentile intervals from a scene-level bootstrap.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import decompose  # noqa: E402
from scalehint import data, results  # noqa: E402
from scalehint.metrics import mra  # noqa: E402

TASKS = ["object_abs_distance", "object_size_estimation", "room_size_estimation"]
SHORT = {"object_abs_distance": "dist", "object_size_estimation": "size", "room_size_estimation": "room"}
MODELS = {"4b": "results", "8b": "results_8b"}
SPLIT = data.load_split("splits/scenes.json")
EV, TR = set(SPLIT["eval"]), set(SPLIT["train"])
TEACH = json.load(open("cache/teacher_scale.json"))
RNG = np.random.default_rng(0)


def rows(cond, rdir):
    df = results.load(cond, rdir)
    return {} if df.empty else {r["qid"]: r for r in df.to_dict("records")}


def ok(r):
    return r is not None and r["pred"] == r["pred"] and r["pred"] is not None and r["pred"] > 0


def boot(items, stat, n_boot):
    """items: list of dicts with 'scene_id'. Resample scenes with replacement."""
    by = {}
    for it in items:
        by.setdefault(it["scene_id"], []).append(it)
    sc = list(by)
    vals = []
    for _ in range(n_boot):
        pick = RNG.choice(len(sc), len(sc), replace=True)
        vals.append(stat([it for i in pick for it in by[sc[i]]]))
    v = np.array([x for x in vals if x == x])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def compression_bins(B):
    out = {}
    for t in TASKS:
        rs = [r for r in B.values() if r["task"] == t and ok(r)]
        g = np.array([r["gt"] for r in rs]); p = np.array([r["pred"] for r in rs])
        edges = np.quantile(g, np.linspace(0, 1, 6))
        bins = []
        for i in range(5):
            m = (g >= edges[i]) & (g <= edges[i + 1])
            bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "gt_med": float(np.median(g[m])),
                         "ratio_med": float(np.median(p[m] / g[m])),
                         "ratio_q25": float(np.percentile(p[m] / g[m], 25)),
                         "ratio_q75": float(np.percentile(p[m] / g[m], 75))})
        vals, cnt = np.unique(p, return_counts=True)
        top5 = float(np.sort(cnt)[-5:].sum() / len(p))
        out[t] = {"bins": bins, "corr_log": float(np.corrcoef(np.log(p), np.log(g))[0, 1]),
                  "n_distinct": int(len(vals)), "top5_share": top5, "n": len(p)}
    return out


def error_budget(B):
    """Variance of e = log(pred/gt) split into: bias (explained by a + b log gt),
    scene-shared residual (ICC of residuals x residual variance), per-question rest."""
    out = {}
    for t in TASKS:
        rs = [r for r in B.values() if r["task"] == t and ok(r)]
        e = np.array([np.log(r["pred"] / r["gt"]) for r in rs]); lg = np.log([r["gt"] for r in rs])
        X = np.column_stack([np.ones_like(lg), lg])
        w = np.linalg.lstsq(X, e, rcond=None)[0]
        res = e - X @ w
        vt, vr = e.var(), res.var()
        d = {"bias": float(1 - vr / vt)}
        import pandas as pd
        df = pd.DataFrame({"scene_id": [r["scene_id"] for r in rs], "r": res})
        if df.groupby("scene_id").size().max() >= 2:
            icc = max(decompose.icc1(df, "r"), 0.0)
            d["scene"] = float(icc * vr / vt)
            d["question"] = float((1 - icc) * vr / vt)
        else:
            d["scene"] = None
            d["question"] = float(vr / vt)
        out[t] = d
    return out


def paired_delta(C, B, n_boot):
    out = {}
    for t in TASKS:
        it = [{"scene_id": C[q]["scene_id"], "d": C[q]["mra"] - B[q]["mra"]}
              for q in C if C[q]["task"] == t and q in B]
        if not it:
            continue
        m = lambda xs: float(np.mean([x["d"] for x in xs]))
        out[t] = {"n": len(it), "delta": m(it), "ci": boot(it, m, n_boot)}
    return out


def elasticity(src, rdir, n_boot):
    ks = [0.5, 0.71, 1.0, 1.41, 2.0]
    F = {k: rows(src if k == 1.0 else f"{src}_k{k:g}", rdir) for k in ks}
    F = {k: f for k, f in F.items() if f}
    if len(F) < 3 or 0.5 not in F or 2.0 not in F:
        return None
    out = {"k_levels": sorted(F)}
    for t in TASKS:
        items = []
        for q, r in F[1.0].items() if 1.0 in F else F[0.5].items():
            if r["task"] != t:
                continue
            pts = [(k, F[k][q]["pred"]) for k in F if q in F[k] and ok(F[k][q])]
            if len(pts) < 3 or not (q in F[0.5] and q in F[2.0] and ok(F[0.5][q]) and ok(F[2.0][q])):
                continue
            lk, lp = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
            items.append({"scene_id": r["scene_id"], "beta": float(np.polyfit(lk, lp, 1)[0]),
                          "dir": float(np.sign(np.log(F[2.0][q]["pred"] / F[0.5][q]["pred"])))})
        mb = lambda xs: float(np.mean([x["beta"] for x in xs]))
        out[t] = {"n": len(items), "beta_mean": mb(items), "beta_mean_ci": boot(items, mb, n_boot),
                  "beta_median": float(np.median([x["beta"] for x in items])),
                  "up": float(np.mean([x["dir"] > 0 for x in items])),
                  "down": float(np.mean([x["dir"] < 0 for x in items])),
                  "same": float(np.mean([x["dir"] == 0 for x in items]))}
        # mean log(pred_k / pred_baseline-of-source-k1) per k, for the figure
        ref = F[1.0] if 1.0 in F else None
        if ref:
            out[t]["curve"] = {f"{k:g}": float(np.mean([np.log(F[k][q]["pred"] / ref[q]["pred"])
                                                        for q in F[k] if q in ref and F[k][q]["task"] == t
                                                        and ok(F[k][q]) and ok(ref[q])])) for k in F}
    return out


def fit_eval(rs, feats, n_boot, items_only=False):
    """Log-linear calibration fit on TRAIN scenes, MRA on EVAL scenes (bootstrap over eval scenes)."""
    tr = [r for r in rs if r["scene_id"] in TR]; ev = [r for r in rs if r["scene_id"] in EV]
    X = lambda xs: np.column_stack([np.ones(len(xs))] + [np.log(np.array([f(r) for r in xs])) for f in feats])
    w = np.linalg.lstsq(X(tr), np.log([r["gt"] for r in tr]), rcond=None)[0]
    pe = np.exp(X(ev) @ w)
    items = [{"scene_id": r["scene_id"], "m": mra(p, r["gt"]), "raw": r["mra"]} for r, p in zip(ev, pe)]
    if items_only:
        return items
    mm = lambda xs: float(np.mean([x["m"] for x in xs]))
    gain = lambda xs: float(np.mean([x["m"] - x["raw"] for x in xs]))
    return {"n": len(items), "mra": mm(items), "mra_ci": boot(items, mm, n_boot),
            "gain_vs_raw": gain(items), "gain_ci": boot(items, gain, n_boot),
            "raw": float(np.mean([x["raw"] for x in items]))}


def headroom(B, n_boot):
    out = {}
    room = [r for r in B.values() if r["task"] == "room_size_estimation" and ok(r)]
    dist = [r for r in B.values() if r["task"] == "object_abs_distance" and ok(r)]
    sext = lambda r: TEACH[r["scene_id"]]["s_ext"]; smed = lambda r: TEACH[r["scene_id"]]["s_med"]
    pred = lambda r: r["pred"]
    for name, rs, cue in [("room", room, sext), ("dist", dist, smed)]:
        out[name] = {"cue_only": fit_eval(rs, [cue], n_boot), "vlm_recal": fit_eval(rs, [pred], n_boot),
                     "vlm_plus_cue": fit_eval(rs, [pred, cue], n_boot)}
        # paired: does the cue add information beyond recalibrating the answer?
        a_ = fit_eval(rs, [pred], 0, items_only=True)
        b_ = fit_eval(rs, [pred, cue], 0, items_only=True)
        it = [{"scene_id": x["scene_id"], "d": y["m"] - x["m"]} for x, y in zip(a_, b_)]
        md = lambda xs: float(np.mean([x["d"] for x in xs]))
        out[name]["cue_over_recal"] = {"delta": md(it), "ci": boot(it, md, n_boot)}
    # oracle use of the anchor: rescale by (true anchor size / model's own estimate of it)
    qs = data.load_questions("data/questions.jsonl")
    A = data.anchor_table(qs)
    size_q = {(q["scene_id"], data.size_object(q["question"])): q["qid"]
              for q in qs if q["task"] == "object_size_estimation"}
    out["anchor_oracle"] = {}
    for t in TASKS:
        items = []
        for r in B.values():
            if r["task"] != t or r["scene_id"] not in EV or r["qid"] not in A or not ok(r):
                continue
            obj, cm = A[r["qid"]]
            aq = size_q.get((r["scene_id"], obj))
            if aq in B and ok(B[aq]):
                items.append({"scene_id": r["scene_id"], "raw": r["mra"],
                              "m": mra(r["pred"] * cm / B[aq]["pred"], r["gt"])})
        mm = lambda xs: float(np.mean([x["m"] for x in xs]))
        gain = lambda xs: float(np.mean([x["m"] - x["raw"] for x in xs]))
        out["anchor_oracle"][t] = {"n": len(items), "raw": float(np.mean([x["raw"] for x in items])),
                                   "mra": mm(items), "gain": gain(items), "gain_ci": boot(items, gain, n_boot)}
    return out


def change_rate(C, B):
    return {t: float(np.mean([C[q]["pred"] != B[q]["pred"] for q in C if C[q]["task"] == t and q in B]))
            for t in TASKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    N = {}
    for m, rdir in MODELS.items():
        B = rows("baseline", rdir)
        if not B:
            continue
        Bdf = results.load("baseline", rdir)
        N[m] = {"n_baseline": len(B)}
        N[m]["decomp"] = {t: decompose.analyse(g, a.n_boot) for t, g in Bdf.groupby("task")}
        N[m]["cross_task"] = decompose.cross_task(Bdf)
        N[m]["compression"] = compression_bins(B)
        N[m]["budget"] = error_budget(B)
        N[m]["delta_k1"] = {}
        N[m]["changed"] = {}
        N[m]["mra"] = {}
        for c in results.conditions(rdir):
            if c == "baseline":
                continue
            C = rows(c, rdir)
            N[m]["delta_k1"][c] = paired_delta(C, B, a.n_boot)
            N[m]["changed"][c] = change_rate(C, B)
            N[m]["mra"][c] = {t: float(np.mean([r["mra"] for r in C.values() if r["task"] == t]))
                              for t in TASKS if any(r["task"] == t for r in C.values())}
        N[m]["mra"]["baseline_eval"] = {t: float(np.mean([r["mra"] for r in B.values()
                                                          if r["task"] == t and r["scene_id"] in EV]))
                                        for t in TASKS}
        N[m]["elasticity"] = {s: elasticity(s, rdir, a.n_boot) for s in ["anchor", "teacher", "answer"]}
        if m == "4b":
            N[m]["headroom"] = headroom(B, a.n_boot)
        print(m, "done", flush=True)
    with open("paper/numbers.json", "w") as f:
        json.dump(N, f, indent=1, default=float)
    appendix_table(N)
    figures(N)


def appendix_table(N):
    """LaTeX rows for the appendix: paired MRA change vs baseline with CI, every condition."""
    order = ["placebo", "spatialprompt", "shuffled", "learned", "teacher_ext",
             "teacher_k0.5", "teacher_k0.71", "teacher", "teacher_k1.41", "teacher_k2",
             "anchor_k0.5", "anchor_k0.71", "anchor", "anchor_k1.41", "anchor_k2",
             "answer_k0.5", "answer", "answer_k2"]
    nice = lambda c: (c.replace("teacher_ext", "depth + extent").replace("_k", " $k$=")
                      .replace("teacher", "depth").replace("spatialprompt", "reasoning"))
    f = lambda d: (f"{100 * d['delta']:+.1f} [{100 * d['ci'][0]:.1f}, {100 * d['ci'][1]:.1f}]"
                   .replace("+", "$+$").replace("-", "$-$"))
    lines = []
    for m in N:
        lines.append(f"\\midrule\n\\multicolumn{{4}}{{l}}{{\\emph{{Qwen3-VL-{m.upper()}}}, baseline MRA: "
                     + ", ".join(f"{100 * N[m]['mra']['baseline_eval'][t]:.1f}" for t in TASKS) + "} \\\\")
        for c in order:
            D = N[m]["delta_k1"].get(c)
            if not D:
                continue
            lines.append(nice(c) + " & " + " & ".join(f(D[t]) if t in D else "--" for t in TASKS) + " \\\\")
    open("paper/appendix_rows.tex", "w").write("\n".join(lines) + "\n")


def figures(N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # LaTeX-rendered text -> embedded Type 1 fonts (AAAI forbids Type 3 and CID/Identity-H fonts)
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
                         "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}"})
    xt = {"object_abs_distance": ([0.5, 1, 2, 5], "ground truth (m)"),
          "object_size_estimation": ([20, 50, 100, 200, 400], "ground truth (cm)"),
          "room_size_estimation": ([5, 10, 20, 50], r"ground truth (m$^2$)")}
    names = {"object_abs_distance": "Object distance", "object_size_estimation": "Object size",
             "room_size_estimation": "Room size"}
    sty = {"4b": dict(color="0.0", marker="o", ls="-"), "8b": dict(color="0.55", marker="s", ls="--")}
    lab = {"4b": "Qwen3-VL-4B", "8b": "Qwen3-VL-8B"}
    # Figure: compression
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.9))
    for i, t in enumerate(TASKS):
        for m in N:
            b = N[m]["compression"][t]["bins"]
            x = [z["gt_med"] for z in b]
            ax[i].plot(x, [z["ratio_med"] for z in b], label=lab[m], ms=3, lw=1, **sty[m])
            if m == "4b":
                ax[i].fill_between(x, [z["ratio_q25"] for z in b], [z["ratio_q75"] for z in b],
                                   color="0.85", lw=0)
        ax[i].axhline(1, color="0.3", lw=0.6, ls=":")
        ax[i].set_xscale("log"); ax[i].set_yscale("log")
        ax[i].set_title(names[t]); ax[i].set_xlabel(xt[t][1] + ", bin median")
        ax[i].set_xticks(xt[t][0]); ax[i].set_xticklabels([f"{v:g}" for v in xt[t][0]])
        ax[i].set_yticks([0.5, 0.7, 1, 1.5, 2]); ax[i].set_yticklabels(["0.5", "0.7", "1", "1.5", "2"])
        ax[i].minorticks_off()
    ax[0].set_ylabel("prediction / truth")
    ax[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig("paper/fig_compression.pdf")
    # Figure: elasticity curves
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.9))
    srcs = {"answer": dict(color="0.0", marker="^"), "teacher": dict(color="0.35", marker="o"),
            "anchor": dict(color="0.6", marker="s")}
    slab = {"answer": "answer (control)", "teacher": "depth scale", "anchor": "anchor object"}
    for i, t in enumerate(TASKS):
        lk = np.log([0.5, 2]); ax[i].plot([0.5, 2], np.exp(lk) / 1, color="0.3", lw=0.6, ls=":")
        for s, st in srcs.items():
            for m in N:
                E = N[m]["elasticity"].get(s)
                if not E or t not in E or "curve" not in E[t]:
                    continue
                c = E[t]["curve"]; ks = sorted(float(k) for k in c)
                ax[i].plot(ks, [np.exp(c[f"{k:g}"]) for k in ks], ms=3, lw=1,
                           ls="-" if m == "4b" else "--", label=f"{slab[s]}" if m == "4b" else None, **st)
        ax[i].set_xscale("log"); ax[i].set_yscale("log")
        ax[i].set_xticks([0.5, 1, 2]); ax[i].set_xticklabels(["0.5", "1", "2"])
        ax[i].set_yticks([0.5, 0.7, 1, 1.5, 2]); ax[i].set_yticklabels(["0.5", "0.7", "1", "1.5", "2"])
        ax[i].minorticks_off()
        ax[i].set_title(names[t]); ax[i].set_xlabel("cue multiplier k")
    ax[0].set_ylabel("answer / answer at k=1")
    ax[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig("paper/fig_elasticity.pdf")


if __name__ == "__main__":
    main()
