"""Step 0: questions, scene split, 16 frames per scene. CPU + network only.

    python prepare_data.py                    # everything
    python prepare_data.py --skip_frames      # questions + split only

Video zips are downloaded one at a time from the HF dataset repo, the needed
mp4s are streamed out, frames are written, and the zip is deleted (unless
--keep_zips). Deterministic: running it on two machines gives the same frames.
"""

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from collections import Counter

from scalehint import ROOT_DEFAULTS, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_repo", default="nyu-visionx/VSI-Bench")
    ap.add_argument("--config", default="full")
    ap.add_argument("--questions", default=ROOT_DEFAULTS["questions"])
    ap.add_argument("--split", default=ROOT_DEFAULTS["split"])
    ap.add_argument("--frames", default=ROOT_DEFAULTS["frames"])
    ap.add_argument("--download_dir", default=None,
                    help="where zips go (default: a temp dir; on Kaggle use /kaggle/tmp)")
    ap.add_argument("--skip_frames", action="store_true")
    ap.add_argument("--keep_zips", action="store_true")
    a = ap.parse_args()

    # questions
    if os.path.exists(a.questions):
        qs = data.load_questions(a.questions)
        print(f"questions: {len(qs)} (cached in {a.questions})")
    else:
        qs = data.build_questions(a.hf_repo, a.config)
        os.makedirs(os.path.dirname(a.questions), exist_ok=True)
        with open(a.questions, "w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q) + "\n")
        print(f"questions: {len(qs)} -> {a.questions}")
    print("  per task:", dict(Counter(q["task"] for q in qs)))

    # split
    ds_of = {q["scene_id"]: q["dataset"] for q in qs}
    if os.path.exists(a.split):
        sp = data.load_split(a.split)
        print(f"split: cached ({len(sp['train'])} train / {len(sp['eval'])} eval scenes)")
    else:
        sp = data.make_split(sorted(ds_of), ds_of)
        os.makedirs(os.path.dirname(a.split), exist_ok=True)
        with open(a.split, "w") as f:
            json.dump(sp, f, indent=1)
        print(f"split: {len(sp['train'])} train / {len(sp['eval'])} eval scenes -> {a.split}")
    assert not set(sp["train"]) & set(sp["eval"]), "split leaks scenes"

    if a.skip_frames:
        return

    from huggingface_hub import hf_hub_download

    need = {}
    for sid, d in ds_of.items():
        if len(data.frame_paths(a.frames, sid)) != data.N_FRAMES:
            need.setdefault(d, set()).add(sid.split("__", 1)[1])
    if not need:
        print("frames: all present")
        return

    dl = a.download_dir or tempfile.mkdtemp(prefix="vsi_")
    os.makedirs(dl, exist_ok=True)
    failed = []
    for d, names in sorted(need.items()):
        print(f"[{d}] need {len(names)} scenes; downloading {d}.zip ...", flush=True)
        zp = hf_hub_download(a.hf_repo, f"{d}.zip", repo_type="dataset", local_dir=dl)
        with zipfile.ZipFile(zp) as z:
            members = {}
            for m in z.namelist():
                if m.lower().endswith(".mp4"):
                    stem = os.path.splitext(os.path.basename(m))[0]
                    members[stem] = m
            for i, name in enumerate(sorted(names)):
                sid = data.scene_key(d, name)
                if name not in members:
                    failed.append((sid, "video not in zip"))
                    continue
                tmp = os.path.join(dl, "_cur.mp4")
                with z.open(members[name]) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                try:
                    data.extract_frames(tmp, os.path.join(a.frames, sid))
                except Exception as e:
                    failed.append((sid, repr(e)))
                os.remove(tmp)
                if (i + 1) % 25 == 0:
                    print(f"  {i + 1}/{len(names)}", flush=True)
        if not a.keep_zips:
            os.remove(zp)
    if failed:
        with open(os.path.join(os.path.dirname(a.frames), "frame_failures.json"), "w") as f:
            json.dump(failed, f, indent=1)
        print(f"frames: {len(failed)} scenes failed (see frame_failures.json)")
    print("frames: done")


if __name__ == "__main__":
    main()
