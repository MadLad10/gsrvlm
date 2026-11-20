# Is Scale All You're Missing? Diagnosing metric errors in frozen VLMs

Code for the GSR-VLM workshop paper. The study plan is in [PLAN.md](PLAN.md) (§0 lists the changes made after the prior-work check).

All GPU work runs on Kaggle or Colab. Two sessions run in parallel:

| Session | Notebook | GPU | Does |
|---|---|---|---|
| A | `notebooks/vlm.ipynb` | Kaggle 2×T4 (or Colab) | data, baseline, decomposition, hint runs, predictor, analysis |
| B | `notebooks/teacher.ipynb` | Colab T4 (or a 2nd Kaggle) | MapAnything metric scale → `cache/teacher_scale.json` |

Handoff: B writes `cache/teacher_scale.json` → download it, commit and push it → `git pull` in A (the cell is already in the notebook).

## Quick start

1. Push this repo to GitHub. Set `REPO_URL` in the first cell of both notebooks.
2. Kaggle: New Notebook → File → Import Notebook → `notebooks/vlm.ipynb`. Accelerator **GPU T4 x2**, Internet **on**.
3. Colab: open `notebooks/teacher.ipynb`, Runtime → T4 GPU. On Colab, the notebooks keep the repo on Google Drive so a disconnect loses nothing.
4. Run the cells top to bottom. Every GPU script appends results per item and resumes, so after a crash you just rerun the cell.

Kaggle interactive sessions lose `/kaggle/working` when they end. The last cell zips `results/`, `figures/` and the small caches into `results_bundle.zip`. Download it whenever a block finishes.

## Steps and outputs

| Step | Command | Output |
|---|---|---|
| data | `python prepare_data.py --download_dir /tmp/vsi` | `data/questions.jsonl`, `splits/scenes.json` (60/40 scene-disjoint, seed 0, stratified by source), `cache/frames/<scene>/00..15.jpg` |
| E0/E1 | `SPLIT=all bash run_conds.sh baseline` | `results/baseline.s<k>.jsonl` |
| E2 | `python decompose.py` | `results/decomposition.json` + GO/NO-GO print |
| E3 | `python run_teacher.py` then `--merge` | `cache/teacher_scale.json` (`s_med`, `s_ext`) |
| E4 | `bash run_conds.sh placebo anchor teacher shuffled ...` | `results/<cond>.s<k>.jsonl` |
| E5 | `python elasticity.py` | `results/elasticity.json`, `figures/elasticity_*.pdf` |
| E6 | `python cache_vis.py`, `python train_predictor.py`, `bash run_conds.sh learned` | `cache/qwen_vis/*.pt`, `cache/learned_scale.json`, `results/predictor.json` |
| tables | `python make_tables.py` | `results/tables.md`, `results/tables.tex` |

`run_conds.sh` starts one process per visible GPU, each running one shard. Logs go to `logs/<cond>.s<k>.log`.

Conditions (see `scalehint/prompts.py`): `baseline`, `placebo`, `spatialprompt`, `teacher`, `teacher_k<k>`, `teacher_ext`, `shuffled`, `learned`, `anchor`, `anchor_k<k>`, with k ∈ {0.5, 0.71, 1.41, 2}.

Result row: `{qid, scene_id, task, gt, raw_answer, pred, unit_seen, mra, condition, k, hint_value, anchor_obj, prompt, time_s}`. Each shard also writes `<cond>.s<k>.meta.json` with the GPU, dtype, torch/transformers versions, frame count and image size.

## Setup details and deviations

- **VLM:** `Qwen/Qwen3-VL-4B-Instruct` via `AutoModelForImageTextToText`, greedy decoding, SDPA attention. Use `--model` to swap it, e.g. `Qwen/Qwen3-VL-2B-Instruct`. The 16 frames go in as 16 images with a 448 px longest side (`--vlm_side`), not as a video. The prompt is the question plus VSI-Bench's postfix ("Please answer the question using a single word or phrase."), with the hint line prepended.
- **dtype on T4:** `--dtype auto` uses fp16 on Turing and bf16 on Ampere+. Before running, the script checks that a forward pass gives finite logits. If fp16 fails that check, it reloads in bf16 (emulated on T4: slower but correct) and logs this in the meta file.
- **Parsing** (`scalehint/parse.py`): the first number wins. A range becomes its midpoint. For `spatialprompt`, only the text after "Final answer" is parsed. An explicit unit is converted (cm↔m, ft, in, sq ft). Unparseable answers score MRA 0, and the rate is reported per condition.
- **MRA:** thresholds 0.50:0.05:0.95, scored correct when |pred−gt|/gt ≤ 1−θ (as in the official code).
- **Teacher:** `facebook/map-anything-apache` in bf16 with the Turing patches from ThriftSplat (`teacher_patches.py`). Frames are resized to a 336 px longest side; on OOM it falls back to 8 frames and records that. No depth GT is used; `make_tables.py` reports the rank correlation with VSI-Bench room-size GT as a sanity check.
- **Anchor hint:** the GT longest dimension of a different object in the same scene, taken from VSI-Bench's object-size questions. The anchor object is never named in the target question. Coverage is limited to scenes that have such questions, and Table 1 deltas are paired on the same question IDs.
- **Predictor target:** log of the teacher's `s_med`, so no benchmark GT is used. It trains on train scenes only, and `train_predictor.py` asserts the split has no overlap.
- **Bootstrap:** 1000 resamples over scenes; a scene drawn twice counts as two scenes.

## Layout

```
scalehint/        parse, metrics, data (frames/split/anchors), prompts, vlm wrapper, results loader
prepare_data.py   run_vlm.py   run_conds.sh   run_teacher.py   teacher_patches.py   install_teacher.sh
cache_vis.py      train_predictor.py
decompose.py      elasticity.py   make_tables.py
notebooks/        vlm.ipynb  teacher.ipynb
```
