"""Run the frozen VLM on one condition. Appends one JSON line per question and
resumes from whatever is already on disk (Kaggle/Colab sessions die).

    python run_vlm.py --condition baseline --split all
    CUDA_VISIBLE_DEVICES=1 python run_vlm.py --condition teacher_k2 --shard 1 --nshards 2

Output: results/<condition>.s<shard>.jsonl  (analysis reads all shards).
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from scalehint import ROOT_DEFAULTS, TASKS, data, prompts
from scalehint.metrics import mra
from scalehint.parse import parse_answer


def done_qids(results_dir, cond):
    done = set()
    for p in glob.glob(os.path.join(results_dir, f"{cond}.s*.jsonl")):
        done |= {r["qid"] for r in data.read_jsonl(p)}
    return done


def derangement(keys, seed=0):
    rng = np.random.default_rng(seed)
    keys = sorted(keys)
    while True:
        perm = rng.permutation(len(keys))
        if all(perm[i] != i for i in range(len(keys))):
            return {keys[i]: keys[perm[i]] for i in range(len(keys))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    ap.add_argument("--split", default="eval", choices=["all", "train", "eval"])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit_per_task", type=int, default=0, help="0 = all")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--vlm_side", type=int, default=448, help="longest image side fed to the VLM")
    ap.add_argument("--max_new_tokens", type=int, default=0, help="0 = 32, or 512 for spatialprompt")
    ap.add_argument("--questions", default=ROOT_DEFAULTS["questions"])
    ap.add_argument("--split_file", default=ROOT_DEFAULTS["split"])
    ap.add_argument("--frames", default=ROOT_DEFAULTS["frames"])
    ap.add_argument("--teacher", default=ROOT_DEFAULTS["teacher"])
    ap.add_argument("--learned", default=ROOT_DEFAULTS["learned"])
    ap.add_argument("--out_dir", default=ROOT_DEFAULTS["results"])
    a = ap.parse_args()

    src, k = prompts.parse_condition(a.condition)
    tasks = [t.strip() for t in a.tasks.split(",") if t.strip()]
    qs = [q for q in data.load_questions(a.questions) if q["task"] in tasks]
    if a.split != "all":
        keep = set(data.load_split(a.split_file)[a.split])
        qs = [q for q in qs if q["scene_id"] in keep]
    if a.limit_per_task:
        rng = np.random.default_rng(0)
        out = []
        for t in tasks:
            tq = [q for q in qs if q["task"] == t]
            idx = sorted(rng.permutation(len(tq))[:a.limit_per_task])
            out += [tq[i] for i in idx]
        qs = out

    # per-question hint values
    hint, extent, anchor_obj = {}, {}, {}
    if src in ("teacher", "teacher_ext", "shuffled"):
        with open(a.teacher) as f:
            T = json.load(f)
        T = {s: v for s, v in T.items() if v.get("s_med")}
        qs = [q for q in qs if q["scene_id"] in T]
        if src == "shuffled":
            # derangement over every scene with a teacher value, independent of
            # sharding / task filters, so all shards agree
            mp = derangement(T.keys())
            for q in qs:
                hint[q["qid"]] = T[mp[q["scene_id"]]]["s_med"]
        else:
            for q in qs:
                hint[q["qid"]] = T[q["scene_id"]]["s_med"] * k
                extent[q["qid"]] = T[q["scene_id"]].get("s_ext")
            if src == "teacher_ext":
                qs = [q for q in qs if extent[q["qid"]]]
    elif src == "learned":
        with open(a.learned) as f:
            L = json.load(f)
        qs = [q for q in qs if q["scene_id"] in L]
        for q in qs:
            hint[q["qid"]] = L[q["scene_id"]] * k
    elif src == "anchor":
        A = data.anchor_table(data.load_questions(a.questions))
        qs = [q for q in qs if q["qid"] in A]
        for q in qs:
            anchor_obj[q["qid"]], cm = A[q["qid"]]
            hint[q["qid"]] = cm * k

    qs = [q for q in qs if len(data.frame_paths(a.frames, q["scene_id"])) == data.N_FRAMES]
    qs.sort(key=lambda q: (q["scene_id"], q["qid"]))
    qs = [q for i, q in enumerate(qs) if i % a.nshards == a.shard]
    done = done_qids(a.out_dir, a.condition)
    todo = [q for q in qs if q["qid"] not in done]
    print(f"{a.condition}: {len(qs)} questions in shard {a.shard}/{a.nshards}, "
          f"{len(qs) - len(todo)} done, {len(todo)} to run", flush=True)
    if not todo:
        return

    import torch
    from scalehint import vlm

    dtype = vlm.pick_dtype(a.dtype)
    model, proc = vlm.load(a.model, dtype)
    f0 = vlm.load_frames(data.frame_paths(a.frames, todo[0]["scene_id"]), a.vlm_side)
    if not vlm.finite_check(model, proc, f0, todo[0]["question"]):
        if dtype == torch.float16:
            print("fp16 gave non-finite logits on this GPU; reloading in bfloat16", flush=True)
            del model
            torch.cuda.empty_cache()
            dtype = torch.bfloat16
            model, proc = vlm.load(a.model, dtype)
            if not vlm.finite_check(model, proc, f0, todo[0]["question"]):
                raise SystemExit("bf16 also non-finite; stop and investigate")
        else:
            raise SystemExit(f"non-finite logits in {dtype}; stop and investigate")

    os.makedirs(a.out_dir, exist_ok=True)
    info = vlm.env_info(a.model, dtype)
    info.update(condition=a.condition, n_frames=data.N_FRAMES, vlm_side=a.vlm_side,
                shard=a.shard, nshards=a.nshards)
    with open(os.path.join(a.out_dir, f"{a.condition}.s{a.shard}.meta.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(info, flush=True)

    out_path = os.path.join(a.out_dir, f"{a.condition}.s{a.shard}.jsonl")
    cur_scene, frames = None, None
    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for i, q in enumerate(todo):
            if q["scene_id"] != cur_scene:
                frames = vlm.load_frames(data.frame_paths(a.frames, q["scene_id"]), a.vlm_side)
                cur_scene = q["scene_id"]
            h = hint.get(q["qid"])
            prompt, reasoning = prompts.build_prompt(
                q["question"], src, hint=h, extent=extent.get(q["qid"]),
                anchor_obj=anchor_obj.get(q["qid"]))
            mnt = a.max_new_tokens or (512 if reasoning else 32)
            t0 = time.time()
            raw = vlm.answer(model, proc, frames, prompt, max_new_tokens=mnt)
            pred, unit = parse_answer(raw, TASKS[q["task"]])
            rec = {
                "qid": q["qid"], "scene_id": q["scene_id"], "task": q["task"],
                "gt": q["gt"], "raw_answer": raw, "pred": pred, "unit_seen": unit,
                "mra": mra(pred, q["gt"]), "condition": a.condition, "k": k,
                "hint_value": h, "anchor_obj": anchor_obj.get(q["qid"]),
                "prompt": prompt, "time_s": round(time.time() - t0, 3),
            }
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t_start
                print(f"  {i + 1}/{len(todo)}  {el / (i + 1):.2f} s/q  "
                      f"eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
