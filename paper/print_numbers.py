"""Print the numbers quoted in paper.tex from paper/numbers.json (cwd = repo root)."""
import json

N = json.load(open("paper/numbers.json"))
T = ["object_abs_distance", "object_size_estimation", "room_size_estimation"]
SHORT = {'object_abs_distance': 'dist', 'object_size_estimation': 'size', 'room_size_estimation': 'room'}
p = lambda x: f"{100 * x:.1f}"
ci = lambda c, s=100, d=1: f"[{s * c[0]:.{d}f}, {s * c[1]:.{d}f}]"
for m in N:
    print(f"===== {m}")
    for t in T:
        d = N[m]["decomp"][t]; b = N[m]["budget"][t]; c = N[m]["compression"][t]
        mr = d.get("mra", {})
        loo = (100 * (mr["scene_loo"] - mr["base"])) if mr else None
        print(f"{t[:10]} n={d['n']} slope={d['slope_pred_on_gt']:.2f} {ci(d['slope_pred_on_gt_ci'], 1, 2)} "
              f"rev={d['slope_pred_on_gt_reverse_bound']:.2f} icc_raw={d['icc_raw']:.2f} {ci(d.get('icc_raw_ci', [0, 0]), 1, 2)} "
              f"icc_res={d['icc_resid']:.2f} {ci(d.get('icc_resid_ci', [0, 0]), 1, 2)} "
              f"budget bias={p(b['bias'])} scene={p(b['scene']) if b['scene'] is not None else '-'} q={p(b['question'])} "
              f"LOO={loo if loo is None else round(loo, 1)} global={round(100 * (mr['global'] - mr['base']), 1) if mr else None} "
              f"corr={c['corr_log']:.2f} top5={p(c['top5_share'])} bins={[round(z['ratio_med'], 2) for z in c['bins']]}")
    print("cross", {k: round(v["spearman"], 2) for k, v in N[m]["cross_task"].items()})
    print("baseline eval MRA", {SHORT[t]: p(v) for t, v in N[m]["mra"]["baseline_eval"].items()})
    for c in ["placebo", "shuffled", "teacher", "teacher_ext", "learned", "anchor", "spatialprompt", "answer"]:
        D = N[m]["delta_k1"].get(c)
        if D:
            print(f"  delta {c:13s}", {SHORT[t]: f"{100 * D[t]['delta']:+.1f} {ci(D[t]['ci'])}" for t in D})
    print("  changed", {c: {t[:4]: round(v, 2) for t, v in d.items()} for c, d in N[m]["changed"].items()
                        if c in ("placebo", "shuffled", "teacher", "anchor", "anchor_k2", "learned")})
    for s, E in N[m]["elasticity"].items():
        if E:
            print(f"  elast {s:8s} k={E['k_levels']}",
                  {SHORT[t]: f"{E[t]['beta_mean']:.3f} {ci(E[t]['beta_mean_ci'], 1, 3)} up{p(E[t]['up'])}/dn{p(E[t]['down'])} med{E[t]['beta_median']:.2f}"
                   for t in T if t in E})
    if "headroom" in N[m]:
        H = N[m]["headroom"]
        for k in ["room", "dist"]:
            print(f"  headroom {k}", {n: f"{p(v['mra'])} gain {100 * v['gain_vs_raw']:+.1f} {ci(v['gain_ci'])} raw {p(v['raw'])}"
                                       for n, v in H[k].items() if n != "cue_over_recal"},
                  "cue_over_recal", f"{100 * H[k]['cue_over_recal']['delta']:+.1f} {ci(H[k]['cue_over_recal']['ci'])}")
        print("  anchor oracle", {SHORT[t]: f"{p(v['mra'])} raw {p(v['raw'])} gain {100 * v['gain']:+.1f} {ci(v['gain_ci'])} n={v['n']}"
                                  for t, v in H["anchor_oracle"].items()})

