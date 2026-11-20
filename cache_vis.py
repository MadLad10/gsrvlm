"""E6 step 1: the VLM's own vision tokens, mean-pooled per frame -> [16, D] fp16.

    python cache_vis.py                       # all scenes (train + eval), resumes
    CUDA_VISIBLE_DEVICES=1 python cache_vis.py --shard 1 --nshards 2

Uses model.get_image_features (post-merger tokens, i.e. what the LLM reads).
"""

import argparse
import json
import os

from scalehint import ROOT_DEFAULTS, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--vlm_side", type=int, default=448)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--questions", default=ROOT_DEFAULTS["questions"])
    ap.add_argument("--frames", default=ROOT_DEFAULTS["frames"])
    ap.add_argument("--out", default="cache/qwen_vis")
    a = ap.parse_args()

    scenes = sorted({q["scene_id"] for q in data.load_questions(a.questions)})
    scenes = [s for i, s in enumerate(scenes) if i % a.nshards == a.shard]
    os.makedirs(a.out, exist_ok=True)
    todo = [s for s in scenes if not os.path.exists(os.path.join(a.out, f"{s}.pt"))
            and len(data.frame_paths(a.frames, s)) == data.N_FRAMES]
    print(f"cache_vis: {len(todo)} scenes to run", flush=True)
    if not todo:
        return

    import torch
    from scalehint import vlm

    dtype = vlm.pick_dtype(a.dtype)
    model, proc = vlm.load(a.model, dtype)
    for i, s in enumerate(todo):
        frames = vlm.load_frames(data.frame_paths(a.frames, s), a.vlm_side)
        f = vlm.vision_features(model, proc, frames)
        if not torch.isfinite(f.float()).all() and dtype == torch.float16:
            print("non-finite fp16 features; switching to bfloat16", flush=True)
            dtype = torch.bfloat16
            model = model.to(dtype)
            f = vlm.vision_features(model, proc, frames)
        torch.save(f, os.path.join(a.out, f"{s}.pt"))
        if i == 0:
            info = vlm.env_info(a.model, dtype)
            info.update(feat_shape=list(f.shape), vlm_side=a.vlm_side,
                        api="get_image_features, mean over post-merger tokens per frame")
            print(info, flush=True)
            with open(os.path.join(a.out, f"_meta.s{a.shard}.json"), "w") as fh:
                json.dump(info, fh, indent=1)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
