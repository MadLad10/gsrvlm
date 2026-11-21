"""Qualitative figures (Qwen3-VL-4B), examples chosen by fixed rules, never by hand.

    python paper/make_gallery.py      (cwd = repo root)

fig_qual_compression.pdf : per task, the evaluation question whose ground truth is closest to
    the task's 10th and 90th percentile (small vs. large truth), baseline answer only.
fig_qual_random.pdf      : 9 evaluation questions drawn at random (seed 0, 3 per task) with
    baseline, anchor cue and stated-answer control at k = 0.5, 1, 2.
"""
import json
import os
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.getcwd())
from scalehint import data  # noqa: E402

EV = set(data.load_split("splits/scenes.json")["eval"])
QTEXT = {q["qid"]: " ".join(q["question"].split()) for q in data.load_questions("data/questions.jsonl")}
R = lambda c: {r["qid"]: r for r in map(json.loads, open(f"results/{c}.s0.jsonl", encoding="utf-8"))}
B = R("baseline")
CUE = {f"{s}{k}": R(s if k == "1" else f"{s}_k{k}") for s in ["anchor", "answer"] for k in ["0.5", "1", "2"]}
TASKS = ["object_abs_distance", "object_size_estimation", "room_size_estimation"]
NAME = {"object_abs_distance": "Distance", "object_size_estimation": "Object size",
        "room_size_estimation": "Room size"}
UNIT = {"object_abs_distance": "m", "object_size_estimation": "cm", "room_size_estimation": r"m$^2$"}


def fmt(v, t):
    return f"{v:.0f}" if t == "object_size_estimation" else f"{v:.1f}"


def complete(q):
    return q in B and B[q]["pred"] and all(q in d and d[q]["pred"] for d in CUE.values())


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 7,
                         "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}"})
    return plt


def panel(fig, x0, y0, w, h, q, lines):
    from PIL import Image
    paths = data.frame_paths("cache/frames", B[q]["scene_id"])
    fw = 0.30 * w
    for j, fi in enumerate([0, 5, 10, 15]):
        ax = fig.add_axes([x0 + (j % 2) * fw / 2, y0 + (1 - j // 2) * h / 2 + 0.004, fw / 2 - 0.003, h / 2 - 0.008])
        ax.imshow(Image.open(paths[fi])); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.3)
    tx = fig.add_axes([x0 + fw + 0.01, y0, w - fw - 0.01, h]); tx.axis("off")
    tx.text(0, 0.98, "\n".join(lines), va="top", ha="left", linespacing=1.4)


def compression_fig(plt):
    sel = []
    for t in TASKS:
        qs = [q for q, r in B.items() if r["task"] == t and r["scene_id"] in EV and complete(q)]
        g = np.array([B[q]["gt"] for q in qs])
        for pct in (10, 90):
            target = np.percentile(g, pct)
            sel.append((t, pct, min(qs, key=lambda q: (abs(B[q]["gt"] - target), q))))
    fig = plt.figure(figsize=(7.0, 3.0))
    for i, (t, pct, q) in enumerate(sel):
        r = B[q]; u = UNIT[t]
        col, row = i % 2, i // 2
        ratio = r["pred"] / r["gt"]
        verdict = "over" if ratio > 1.05 else ("under" if ratio < 0.95 else "close to")
        qs = textwrap.wrap(QTEXT[q].replace("&", r"\&"), 50)[:3]
        lines = [rf"\textbf{{{NAME[t]}, {'small' if pct == 10 else 'large'} truth.}}"] + qs + [
            rf"Truth \textbf{{{fmt(r['gt'], t)}}} {u}, answer \textbf{{{fmt(r['pred'], t)}}} {u} "
            rf"($\times${ratio:.2f}, {verdict}{'estimate' if verdict != 'close to' else ' truth'})"]
        panel(fig, 0.0 + col * 0.505, 0.67 - row * 0.335, 0.495, 0.31, q, lines)
    fig.savefig("paper/fig_qual_compression.pdf")
    return sel


def random_fig(plt):
    rng = np.random.default_rng(0)
    sel = []
    for t in TASKS:
        qs = sorted(q for q, r in B.items() if r["task"] == t and r["scene_id"] in EV and complete(q))
        sel += [(t, q) for q in rng.choice(qs, 3, replace=False)]
    fig = plt.figure(figsize=(7.0, 6.3))
    for i, (t, q) in enumerate(sel):
        r = B[q]; u = UNIT[t]; c = {n: d[q] for n, d in CUE.items()}
        qs = textwrap.wrap(QTEXT[q].replace("&", r"\&"), 50)[:3]
        lines = [rf"\textbf{{{NAME[t]}.}}"] + qs + [
            rf"Truth \textbf{{{fmt(r['gt'], t)}}} {u}, baseline \textbf{{{fmt(r['pred'], t)}}} {u}. At $k=0.5, 1, 2$:",
            rf"anchor ({c['anchor1']['anchor_obj']}, {c['anchor1']['hint_value']:.0f} cm): "
            + ", ".join(fmt(c[f'anchor{k}']['pred'], t) for k in ['0.5', '1', '2']),
            "stated answer: " + ", ".join(fmt(c[f'answer{k}']['pred'], t) for k in ['0.5', '1', '2'])]
        col, row = i % 2, i // 2
        panel(fig, col * 0.505, 0.81 - row * 0.2, 0.495, 0.18, q, lines)
    fig.savefig("paper/fig_qual_random.pdf")
    return sel


if __name__ == "__main__":
    plt = setup()
    s1 = compression_fig(plt)
    s2 = random_fig(plt)
    json.dump({"compression": s1, "random": [list(x) for x in s2]}, open("paper/gallery_examples.json", "w"), indent=1)
    for t, pct, q in s1:
        print("comp", t[:10], pct, q, B[q]["gt"], B[q]["pred"])
    for t, q in s2:
        print("rand", t[:10], q, B[q]["gt"], B[q]["pred"])

