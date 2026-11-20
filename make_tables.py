"""Every number in the paper comes from here. Reads results/*.jsonl and the JSON
reports; writes results/tables.md and results/tables.tex.

    python make_tables.py [--n_boot 1000]

Table 1: MRA per task x condition on the eval split, with scene-bootstrap 95% CI,
         paired delta vs baseline (same questions), unparseable rate.
         Row "LOO scene rescale (bound)" = decompose.loo_mras on baseline, eval scenes.
Table 2: decomposition (results/decomposition.json).
Table 3: elasticity (results/elasticity.json).
Extras:  teacher sanity vs ground-truth room size; predictor report.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from decompose import add_resid, loo_mras
from scalehint import ROOT_DEFAULTS, TASKS, data, results
from scalehint.metrics import scene_bootstrap

ORDER = ["baseline", "placebo", "spatialprompt", "shuffled", "learned", "teacher",
         "teacher_ext", "anchor"]
SHORT = {"object_abs_distance": "Abs. dist.", "object_size_estimation": "Obj. size",
         "room_size_estimation": "Room size"}


def fmt(v, ci=None, pct=True):
    if v is None or not np.isfinite(v):
        return "–"
    s = f"{100 * v:.1f}" if pct else f"{v:.2f}"
    if ci and all(np.isfinite(ci)):
        lo, hi = (100 * ci[0], 100 * ci[1]) if pct else ci
        s += f" [{lo:.1f}, {hi:.1f}]" if pct else f" [{lo:.2f}, {hi:.2f}]"
    return s


def table1(n_boot):
    ev = set(data.load_split(ROOT_DEFAULTS["split"])["eval"])
    base = results.load("baseline")
    if base.empty:
        return "Table 1: no baseline results.\n", []
    base = base[base["scene_id"].isin(ev)]
    conds = results.conditions()
    main = [c for c in ORDER if c in conds] + sorted(c for c in conds if c not in ORDER)
    rows = []
    for c in main:
        df = results.load(c)
        df = df[df["scene_id"].isin(ev)]
        for t in TASKS:
            d = df[df["task"] == t]
            if d.empty:
                continue
            b = base[(base["task"] == t) & base["qid"].isin(d["qid"])]
            pair = d.merge(b[["qid", "mra"]], on="qid", suffixes=("", "_b"))
            pair["delta"] = pair["mra"] - pair["mra_b"]
            rows.append({
                "condition": c, "task": t, "n": len(d),
                "mra": d["mra"].mean(),
                "mra_ci": scene_bootstrap(d, lambda x: x["mra"].mean(), n_boot),
                "delta": pair["delta"].mean() if c != "baseline" else np.nan,
                "delta_ci": scene_bootstrap(pair, lambda x: x["delta"].mean(), n_boot)
                if c != "baseline" and len(pair) else None,
                "unparse": 1 - d["ok"].mean(),
            })
    # oracle bound
    for t in TASKS:
        d = base[base["task"] == t]
        if len(d):
            d, _ = add_resid(d)
            m = loo_mras(d)
            if m:
                rows.append({"condition": "LOO scene rescale (bound)", "task": t, "n": m["n"],
                             "mra": m["scene_loo"], "mra_ci": None,
                             "delta": m["scene_loo"] - m["base"], "delta_ci": None,
                             "unparse": np.nan})
    md = ["## Table 1: MRA (x100) on eval scenes; delta is paired vs baseline on the same questions\n",
          "| condition | task | n | MRA [95% CI] | Δ vs base [95% CI] | unparseable |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['condition']} | {SHORT[r['task']]} | {r['n']} | {fmt(r['mra'], r['mra_ci'])} | "
                  f"{fmt(r['delta'], r['delta_ci'])} | {fmt(r['unparse'])} |")
    return "\n".join(md) + "\n", rows


def table2():
    p = os.path.join(ROOT_DEFAULTS["results"], "decomposition.json")
    if not os.path.exists(p):
        return "Table 2: run decompose.py first.\n"
    with open(p) as f:
        D = json.load(f)
    md = [f"## Table 2: error decomposition ({D['condition']}, all scenes)\n",
          "| task | n | slope b | ICC raw [CI] | ICC resid [CI] | MRA base | global | scene LOO | scene LOO resid |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t, r in D["tasks"].items():
        m = r.get("mra", {})
        md.append(f"| {t} | {r['n']} | {fmt(r['slope_b'], pct=False)} | "
                  f"{fmt(r['icc_raw'], r.get('icc_raw_ci'), pct=False)} | "
                  f"{fmt(r['icc_resid'], r.get('icc_resid_ci'), pct=False)} | "
                  f"{fmt(m.get('base'))} | {fmt(m.get('global'))} | {fmt(m.get('scene_loo'))} | "
                  f"{fmt(m.get('scene_loo_res'))} |")
    if D.get("cross_task"):
        md.append("\nCross-task Spearman of scene-mean residuals: " +
                  "; ".join(f"{k}: {v['spearman']:.2f} (n={v['n_scenes']})"
                            for k, v in D["cross_task"].items()))
    if D.get("go_no_go"):
        md.append(f"\nGo/no-go: {D['go_no_go']}")
    return "\n".join(md) + "\n"


def table3():
    p = os.path.join(ROOT_DEFAULTS["results"], "elasticity.json")
    if not os.path.exists(p):
        return "Table 3: run elasticity.py first.\n"
    with open(p) as f:
        E = json.load(f)
    md = ["## Table 3: scale elasticity (beta = d log pred / d log k)\n",
          "| hint | task | n | median beta [CI] | beta<0.2 | beta>0.8 | copy rate @k=1 |",
          "|---|---|---|---|---|---|---|"]
    for src, R in E.items():
        for t, r in R["tasks"].items():
            cr = R["copy_rate"].get("1", {}).get(t, np.nan)
            md.append(f"| {src} | {SHORT.get(t, t)} | {r['n']} | "
                      f"{fmt(r['median_beta'], r['median_beta_ci'], pct=False)} | "
                      f"{fmt(r['frac_ignore(beta<0.2)'])} | {fmt(r['frac_full(beta>0.8)'])} | {fmt(cr)} |")
    return "\n".join(md) + "\n"


def teacher_sanity():
    p = ROOT_DEFAULTS["teacher"]
    if not os.path.exists(p):
        return "Teacher: no cache/teacher_scale.json.\n"
    with open(p) as f:
        T = json.load(f)
    qs = data.load_questions(ROOT_DEFAULTS["questions"])
    room = {}
    for q in qs:
        if q["task"] == "room_size_estimation":
            room[q["scene_id"]] = q["gt"]
    rows = [(np.sqrt(room[s]), v["s_med"], v.get("s_ext"))
            for s, v in T.items() if v.get("s_med") and s in room]
    n_ok = sum(1 for v in T.values() if v.get("s_med"))
    n_err = sum(1 for v in T.values() if v.get("error"))
    n8 = sum(1 for v in T.values() if v.get("note"))
    if len(rows) < 5:
        return f"Teacher: {n_ok} scenes ok, {n_err} errors.\n"
    a = pd.DataFrame(rows, columns=["sqrt_area", "s_med", "s_ext"])
    rho_med = a["sqrt_area"].rank().corr(a["s_med"].rank())
    rho_ext = a["sqrt_area"].rank().corr(a["s_ext"].rank())
    ratio = np.log(a["s_ext"] / a["sqrt_area"])
    return ("## Teacher sanity (no depth GT used; room-size GT as a proxy)\n\n"
            f"- scenes: {n_ok} ok, {n_err} errors, {n8} fell back to 8 frames\n"
            f"- Spearman(s_med, sqrt(room area)) = {rho_med:.2f}; "
            f"Spearman(s_ext, sqrt(room area)) = {rho_ext:.2f} (n={len(a)})\n"
            f"- log(s_ext / sqrt(area)): median {ratio.median():.2f}, IQR "
            f"{ratio.quantile(.25):.2f}..{ratio.quantile(.75):.2f}\n")


def predictor():
    p = os.path.join(ROOT_DEFAULTS["results"], "predictor.json")
    if not os.path.exists(p):
        return "Predictor: run train_predictor.py first.\n"
    with open(p) as f:
        P = json.load(f)
    tf = P["transformer"]
    return ("## Predictor (median |log error| vs teacher on eval scenes)\n\n"
            f"- constant {P['constant']['eval']:.3f} | ridge {P['ridge']['eval']:.3f} "
            f"({P['ridge']['params']:,} params) | transformer {tf['eval']:.3f} "
            f"({tf['params']:,} params, {tf['train_s_total_3_seeds']} s for 3 seeds on "
            f"{tf['gpu'] or 'CPU'}, peak {tf['peak_vram_gb']} GB)\n"
            f"- chosen on validation: {P['chosen']}; n_train={P['n_train']}, n_eval={P['n_eval']}\n")


def to_tex(rows):
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Condition & Task & $n$ & MRA & $\Delta$ \\", r"\midrule"]
    for r in rows:
        d = "--" if not np.isfinite(r["delta"]) else f"{100 * r['delta']:+.1f}"
        c = r["condition"].replace("_", r"\_")
        lines.append(f"{c} & {SHORT[r['task']]} & {r['n']} & {100 * r['mra']:.1f} & {d} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    t1, rows = table1(a.n_boot)
    md = "\n".join([t1, table2(), table3(), teacher_sanity(), predictor()])
    out = ROOT_DEFAULTS["results"]
    with open(os.path.join(out, "tables.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out, "tables.tex"), "w", encoding="utf-8") as f:
        f.write(to_tex(rows))
    print(md)


if __name__ == "__main__":
    main()
