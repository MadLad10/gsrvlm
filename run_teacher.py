"""E3: metric scale per scene from MapAnything on the same 16 frames.

    python run_teacher.py                     # all scenes, resumes
    CUDA_VISIBLE_DEVICES=1 python run_teacher.py --shard 1 --nshards 2

Writes cache/teacher_scale.s<k>.json; merge with --merge into
cache/teacher_scale.json:
  s_med  median metric depth over valid pixels of all frames (primary scale)
  s_ext  diagonal of the 5-95 percentile box of the valid point cloud
Needs MapAnything installed (see README: install it in its own session).
"""

import argparse
import glob
import json
import os
import time

import numpy as np

from scalehint import ROOT_DEFAULTS, data


def to_np(x):
    import torch
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def scene_scale(preds):
    zs, pts = [], []
    for p in preds:
        z = to_np(p["depth_z"])[0].squeeze()
        P = to_np(p["pts3d"])[0]
        m = (to_np(p["mask"])[0].squeeze() > 0) if "mask" in p else np.ones(z.shape, bool)
        m &= np.isfinite(z) & (z > 0)
        zs.append(z[m])
        pts.append(P[m])
    z = np.concatenate(zs)
    P = np.concatenate(pts)
    P = P[np.isfinite(P).all(1)]
    lo, hi = np.percentile(P, 5, axis=0), np.percentile(P, 95, axis=0)
    return {"s_med": float(np.median(z)), "s_ext": float(np.linalg.norm(hi - lo)),
            "valid_frac": float(len(z) / sum(np.prod(to_np(p["depth_z"])[0].squeeze().shape)
                                             for p in preds))}


def merge(out_dir):
    allv = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "teacher_scale.s*.json"))):
        with open(p) as f:
            allv.update(json.load(f))
    dst = os.path.join(out_dir, "teacher_scale.json")
    with open(dst, "w") as f:
        json.dump(allv, f, indent=1, sort_keys=True)
    ok = sum(1 for v in allv.values() if v.get("s_med"))
    print(f"merged {len(allv)} scenes ({ok} with s_med) -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="facebook/map-anything-apache")
    ap.add_argument("--dtype", default="auto", help="auto (bf16) | bfloat16 | float32")
    ap.add_argument("--longest_side", type=int, default=336)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--questions", default=ROOT_DEFAULTS["questions"])
    ap.add_argument("--frames", default=ROOT_DEFAULTS["frames"])
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    out_dir = os.path.dirname(ROOT_DEFAULTS["teacher"])
    os.makedirs(out_dir, exist_ok=True)
    if a.merge:
        return merge(out_dir)

    import torch
    import teacher_patches as tp

    scenes = sorted({q["scene_id"] for q in data.load_questions(a.questions)})
    scenes = [s for i, s in enumerate(scenes) if i % a.nshards == a.shard]
    out_path = os.path.join(out_dir, f"teacher_scale.s{a.shard}.json")
    res = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            res = json.load(f)
    todo = [s for s in scenes if (s not in res or res[s].get("error"))
            and len(data.frame_paths(a.frames, s)) == data.N_FRAMES]
    print(f"teacher: {len(scenes)} scenes in shard, {len(todo)} to run", flush=True)
    if not todo:
        return

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    model = tp.load_model(a.repo, dtype)
    gpu = torch.cuda.get_device_name(0)
    for i, s in enumerate(todo):
        paths = data.frame_paths(a.frames, s)
        t0 = time.time()
        rec = {"n_frames": len(paths), "longest_side": a.longest_side,
               "dtype": str(dtype).replace("torch.", ""), "gpu": gpu}
        try:
            torch.cuda.reset_peak_memory_stats()
            try:
                preds = tp.infer(model, paths, a.longest_side)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                paths = paths[::2]
                rec["n_frames"] = len(paths)
                rec["note"] = "OOM at 16 frames; used 8"
                preds = tp.infer(model, paths, a.longest_side)
            rec.update(scene_scale(preds))
            if not np.isfinite(rec["s_med"]):
                rec["error"] = "non-finite depth"
                rec["s_med"] = None
            rec["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        except Exception as e:
            rec["error"] = repr(e)[:300]
            torch.cuda.empty_cache()
        rec["time_s"] = round(time.time() - t0, 2)
        res[s] = rec
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(todo)} {s}: s_med={rec.get('s_med')} "
                  f"s_ext={rec.get('s_ext')} {rec['time_s']}s {rec.get('error', '')}", flush=True)


if __name__ == "__main__":
    main()
