"""Figure 1: one typical example per task (Qwen3-VL-4B), chosen by a fixed rule:
the evaluation question whose baseline |log error| is closest to its task's median,
among questions answered under every depth / anchor / answer condition at k=0.5,1,2.

    python paper/make_teaser.py      (cwd = repo root) -> paper/fig_teaser.pdf
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from scalehint import data  # noqa: E402

EV = set(data.load_split("splits/scenes.json")["eval"])
R = lambda c: {r["qid"]: r for r in map(json.loads, open(f"results/{c}.s0.jsonl", encoding="utf-8"))}
CONDS = {"baseline": "baseline"}
for s in ["teacher", "anchor", "answer"]:
    for k in ["0.5", "1", "2"]:
        CONDS[f"{s}{k}"] = s if k == "1" else f"{s}_k{k}"
D = {name: R(c) for name, c in CONDS.items()}
UNIT = {"object_abs_distance": "m", "object_size_estimation": "cm", "room_size_estimation": r"m$^2$"}
TASKS = ["object_abs_distance", "room_size_estimation"]


def pick(task):
    B = D["baseline"]
    qs = [q for q, r in B.items() if r["task"] == task and r["scene_id"] in EV
          and all(q in d and d[q]["pred"] for d in D.values())]
    err = {q: abs(np.log(B[q]["pred"] / B[q]["gt"])) for q in qs}
    med = np.median([abs(np.log(r["pred"] / r["gt"])) for r in B.values()
                     if r["task"] == task and r["scene_id"] in EV and r["pred"]])
    return min(qs, key=lambda q: (abs(err[q] - med), q))


def main():
    sel = {t: pick(t) for t in TASKS}
    info = {}
    for t, q in sel.items():
        r = D["baseline"][q]
        info[t] = {"qid": q, "scene": r["scene_id"], "question": data.load_questions("data/questions.jsonl"),
                   "gt": r["gt"], "preds": {n: D[n][q]["pred"] for n in D},
                   "hints": {n: D[n][q]["hint_value"] for n in D}, "anchor_obj": D["anchor1"][q]["anchor_obj"]}
        qtext = next(x["question"] for x in info[t]["question"] if x["qid"] == q)
        info[t]["question"] = qtext
        print(t, q, r["scene_id"], "gt", r["gt"], {n: v for n, v in info[t]["preds"].items()},
              "hints", {n: (round(v, 2) if v else v) for n, v in info[t]["hints"].items()}, info[t]["anchor_obj"])
        print("  Q:", qtext)
    json.dump(info, open("paper/teaser_examples.json", "w"), indent=1)
    draw(info)


def draw(info):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 7.5,
                         "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}"})
    import textwrap
    fig = plt.figure(figsize=(7.0, 2.05))
    frames_idx = [0, 5, 10, 15]
    label = {"object_abs_distance": "(a) Object distance", "room_size_estimation": "(b) Room size"}
    for row, t in enumerate(TASKS):
        e = info[t]
        y0 = 0.515 - row * 0.5
        paths = data.frame_paths("cache/frames", e["scene"])
        for j, fi in enumerate(frames_idx):
            ax = fig.add_axes([0.003 + (j % 2) * 0.118, y0 + (1 - j // 2) * 0.235, 0.115, 0.225])
            ax.imshow(Image.open(paths[fi]), aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.3)
        u = UNIT[t]; p = e["preds"]; fmt = (lambda v: f"{v:.1f}")
        q = " ".join(e["question"].split()).replace("&", r"\&")
        qlines = textwrap.wrap(q, 100)
        tx = fig.add_axes([0.25, y0, 0.75, 0.47]); tx.axis("off")
        lines = [rf"\textbf{{{label[t]}.}} " + qlines[0]] + qlines[1:] + [
            rf"Truth: \textbf{{{fmt(e['gt'])}}} {u}. Baseline answer: \textbf{{{fmt(p['baseline'])}}} {u}. "
            r"Answers when the cue is multiplied by $k = 0.5,\ 1,\ 2$:",
            rf"$\bullet$ depth scale ({e['hints']['teacher1']:.1f} m at $k{{=}}1$): "
            rf"{fmt(p['teacher0.5'])}, {fmt(p['teacher1'])}, {fmt(p['teacher2'])} {u}",
            rf"$\bullet$ anchor object ({e['anchor_obj']}, {e['hints']['anchor1']:.0f} cm at $k{{=}}1$): "
            rf"{fmt(p['anchor0.5'])}, {fmt(p['anchor1'])}, {fmt(p['anchor2'])} {u}",
            rf"$\bullet$ stated answer, control ({fmt(e['hints']['answer1'])} {u} at $k{{=}}1$): "
            rf"{fmt(p['answer0.5'])}, {fmt(p['answer1'])}, {fmt(p['answer2'])} {u}"]
        tx.text(0.0, 0.97, "\n".join(lines), va="top", ha="left", linespacing=1.5)
    fig.savefig("paper/fig_teaser.pdf")


if __name__ == "__main__":
    main()
