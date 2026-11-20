# Is Scale All You're Missing? Diagnosing Metric Errors in Frozen VLMs

Plan for code agents. Workshop paper (GSR-VLM), 4 pages. Deadline: tomorrow.
Owner: Wasif. Compute: **all GPU work on Kaggle (2×T4) / Colab (T4)**; no local GPU runs.
Kaggle runs the VLM (`notebooks/vlm.ipynb`); a second session (Colab) runs MapAnything in parallel (`notebooks/teacher.ipynb`).

---

## 0. Changes after the prior-work check (2026-11-20) — these override the sections below

1. **Readout-collapse confound in E2.** If the model answers ~2 m to everything, log(pred/gt) tracks log(gt), which is scene-correlated (big rooms → big distances) and inflates the between-scene share. E2 now also reports the slope b of e on log(gt), the ICC on residuals of e ~ a + b·log(gt), and a *global* (task-level, leave-scene-out) rescale next to the scene LOO rescale. Scene-specific scale = scene-LOO-on-residuals minus global. Report the raw share as an upper bound and the residual share as the lower bound. Also: cross-task Spearman of scene-mean residuals (is the scale error shared across distance / size / room?).
2. **Framing is two-way, not four-way.** The design separates scene-shared error from per-question error. Do not claim to separate grounding vs geometry vs readout (except the slope b, which measures readout collapse).
3. **New E4 conditions.**
   - `spatialprompt`: reference-object reasoning instruction (Liao et al., EMNLP 2024, arXiv 2409.09788): the text-only prior-work baseline. Capped at 100 questions/task (it generates up to 512 tokens).
   - `anchor`, `anchor_k{0.5,0.71,1.41,2}`: ground-truth longest dimension of a *different* object in the same scene (from VSI-Bench's own object-size questions), optionally × k. It is a direct metric cue, so a low elasticity here can't be blamed on the model failing to turn a camera-depth number into a distance. Needs no teacher.
4. **Related work to add:** Consistent Yet Wrong (Bhat & Yamasaki, CVPRW 2026, arXiv 2606.02742), the closest prior work: spatial VLMs give view-stable but wrong answers ("evidence insensitivity"). They vary *views*; we vary an injected *scale cue*. Also SpatialPrompt/Q-Spatial (2409.09788) and UniMetric (2610.16945). Seeing Is Not Measuring finds object size is detector-bound (supports H2's object-size prediction).
5. **Predictor demoted.** ~170 training scenes, so a ridge probe on mean-pooled vision tokens is the main model; the frame-transformer is reported alongside, and the one picked on validation feeds `learned`. Training runs on the notebook GPU (or CPU), not the 2060. It's first to drop if behind.
6. **Frames as images.** The VLM gets the 16 frames as 16 images (448 px longest side), not as a video. Same input for every condition; avoids video-processor version quirks. State this as a deviation from the official VSI-Bench protocol.
7. **MRA uses `<=`** (as in the official VSI-Bench code), not `<`.
8. **T4 dtypes.** VLM: fp16 with an automatic finite-logit check that falls back to bf16. MapAnything: bf16 always (fp16 → all-NaN on Turing, measured in ThriftSplat).

Drop order if behind (replaces §6): predictor/`learned` → `teacher_ext` → k = 0.71/1.41 (both sources) → `spatialprompt`. Never drop E2 or E5.

---

## 1. Research question

### 1.1 The question
When a frozen VLM answers a metric spatial question ("how far is the sofa from the TV?") wrongly, **how much of the error comes from not knowing the scene's absolute scale**, and how much from everything else?

Any metric answer factors as **answer = scale × relative geometry**. A wrong answer has four possible causes:

| Error source | Meaning | Example |
|---|---|---|
| Scale | Wrong idea of how big the scene is | Thinks the whole room is 1.5× larger |
| Geometry | Right scale, wrong relative layout | Places the sofa too close to the wall |
| Grounding | Measures the wrong object or endpoints | Uses the wrong chair |
| Readout | Sloppy or default numbers | Answers "2 m" to everything |

Prior work fixes all four at once (tool pipelines, VGGT/MapAnything fusion, fine-tuned scale tokens) and never measures which one dominates. Tool pipelines move absolute distance from 0.46 to 0.74 MRA (Seeing Is Not Measuring, 2026), but nobody can say *why*. We split the error.

### 1.2 Why it matters
- Tells the field whether to invest in scale estimation (cheap, one number) or in richer 3D representations (expensive).
- Explains the gains of heavier methods.
- Gives a cheap baseline: if one number recovers most of the gap, a 10M-parameter predictor competes with 1B-parameter 3D encoders.

### 1.3 Closest prior work (cite; we differ by keeping the VLM frozen and *measuring*)
- **RoboTracer** (arXiv 2512.13660): `<SCALE>` token + MLP scale decoder on MapAnything encoder; fine-tunes the VLM.
- **Unlocking Dense Metric Depth Estimation in VLMs** (arXiv 2605.15876): depth decoded from the VLM's vision tokens with a light head; fine-tuned.
- **SpatialMind** (arXiv 2610.04139): metric depth adapter for scale.
- **VLM-3R** (arXiv 2505.20279): notes VGGT lacks metric scale; fuses 3D tokens.
- **Seeing Is Not Measuring** (arXiv 2609.29073), **SpaceMCP** (2611.15680): tool-augmented metric reasoning.
- **ReVSI** (arXiv 2604.24300): VSI-Bench label issues; distance skewed to 0–2 m.
- **Spatial Forcing** (2510.12276), **SpatialStack** (2603.27437): VGGT feature alignment/fusion.

None isolates scale as a factor in a frozen VLM, and none measures whether the model *uses* a scale cue (elasticity).

### 1.4 Hypotheses (state these in the paper before results)
- **H1 (decomposition):** A large share of log-error variance is shared within a scene (scene-level scale error).
- **H2 (hint helps where scale matters):** A correct scale hint improves room size and long absolute distances most, object size least (semantic size priors already work), short co-visible distances little (2D cues suffice).
- **H3 (usage):** The VLM's answers scale with the hint (elasticity > 0), but less than 1 (partial use / anchoring to priors).
- **H4 (cheap scale):** A ~5–10M predictor on the VLM's own pooled vision tokens recovers most of the teacher-hint gain.

Any outcome is publishable. If H1/H2 fail: "scale is not the bottleneck for frozen VLMs"; the controls make that credible.

### 1.5 Contributions (target)
1. Error decomposition of frozen-VLM metric answers into scene-level (scale) vs residual components.
2. **Scale elasticity**: a metric for whether a VLM uses a scale cue.
3. Per-task map of where scale matters.
4. A tiny scale predictor trained in minutes on a 6 GB RTX 2060.

---

## 2. Setup

- **VLM:** `Qwen/Qwen3-VL-4B-Instruct` (verify HF id), frozen, fp16 on Kaggle T4. Greedy decoding, temperature 0. If fp16 produces NaN/inf on T4, fall back to fp32 vision tower + fp16 LLM, and log it.
- **Fallback VLM** (if Kaggle unavailable): `Qwen3-VL-2B-Instruct` fp16 locally (~4.5 GB).
- **Teacher:** MapAnything (`facebook/map-anything`, verify id), metric output. Run locally on 2060; reuse ThriftSplat's loading code and the bf16-NaN fix (on 2060 bf16 is emulated, so prefer fp16 if stable, else fp32 for problem layers).
- **Benchmark:** VSI-Bench (`nyu-visionx/VSI-Bench`, verify id and that videos are included). Use ReVSI instead/also if its data is downloadable (cleaner GT). Tasks (verify exact names in the dataset):
  - `object_abs_distance`
  - `object_size_estimation`
  - `room_size_estimation`
- **Frames:** 16 uniformly sampled per video; same frames for VLM and MapAnything.
- **Metric:** VSI-Bench MRA = mean over θ ∈ {0.50, 0.55, …, 0.95} of 1[|pred − gt| / gt < 1 − θ]. Also report median |log(pred/gt)|.
- **Split:** scene-disjoint. 60% of scenes for predictor training, 40% for all evaluation. Fix seed 0; save split as `splits/scenes.json`.

---

## 3. Experiments (in order)

### E0. Go/no-go (≈1 h, Kaggle) — RUN FIRST
1. Baseline Qwen on all eval questions of the 3 tasks (or ~300 per task if slow).
2. Parse a single number from each answer (regex, first float; log unparseable rate).
3. Compute variance decomposition (E2) on absolute distance.
- **Go** if between-scene share ≥ ~0.4 or the leave-one-out rescale (E2) gains ≥ 0.05 MRA.
- **No-go**: stop and report numbers to Wasif before continuing.

### E1. Baseline (part of E0)
Output: `results/baseline.jsonl` with `{qid, scene_id, task, gt, raw_answer, pred, mra}`.

### E2. Error decomposition (no extra GPU)
For each question: e = log(pred / gt).
- Between-scene share = Var(scene means of e) / Var(e) (one-way ANOVA / ICC estimate; only scenes with ≥ 2 questions).
- **Leave-one-out scene rescale:** for question i in scene s, c = mean of e over other questions in s; corrected pred = pred / exp(c). Report MRA after correction = upper bound on what a perfect *learned scene scale* could give.
- Do it per task and pooled; also split absolute distance by GT distance (< 2 m vs ≥ 2 m).
- 1000× scene-level bootstrap for 95% CIs.

### E3. Teacher scale (local 2060)
For each scene, run MapAnything on the 16 frames and save:
- `s_med`: median metric depth over valid pixels of all frames (primary scale).
- `s_ext`: diagonal of the 5–95 percentile bounding box of the point cloud (secondary, for room size).
- Save to `cache/teacher_scale.json`. Log failures/OOM; reduce to 8 frames if needed and record it.
- Sanity check: on scenes with GT available (if ScanNet/ARKit depth is accessible), report teacher scale error. If not, skip and state it.

### E4. Hint runs (Kaggle)
Prepend exactly one line to the question:
`Scale reference: the typical distance from the camera to visible surfaces in this video is about {X:.1f} m.`
Conditions (same frames, same decoding):

| Condition | X |
|---|---|
| `teacher` | s_med |
| `shuffled` | s_med of a different random scene (derangement, seed 0) |
| `placebo` | no number: `Scale reference: consider the physical scale of the scene carefully.` |
| `k{0.5,0.71,1.41,2}` | s_med × k |
| `teacher_ext` (room size only) | add `The scene spans about {s_ext:.1f} m.` |

Outputs: `results/hint_{condition}.jsonl`, same schema as baseline.

### E5. Scale elasticity
For each question, fit slope β of log(pred) vs log(k) over k ∈ {0.5, 0.71, 1, 1.41, 2}.
Report per-task median β with bootstrap CI and the fraction of questions with β ≈ 0 (ignores the hint) vs β ≈ 1 (fully uses it).
Also report **copy rate**: fraction of answers within 5% of the hint value (anchoring check).

### E6. Learned scale predictor (cache on Kaggle, train on 2060)
**Cache (Kaggle):** run only Qwen's vision encoder on the 16 frames per scene (train + eval scenes); mean-pool patch tokens per frame → `[16, D]` fp16. Save `cache/qwen_vis/{scene_id}.pt`. Verify how to call the vision tower in the current transformers API and record it in the README.
**Target:** log(s_med) from E3 (no GT used: self-supervised from the teacher).
**Model:** linear proj D→256, 2-layer transformer encoder over frames (4 heads), mean pool, MLP → scalar. ~2–5M params. Report the exact count.
**Training (2060):** AdamW lr 3e-4, wd 0.05, batch 32, 200 epochs max, early stop on 10% of train scenes held out, fp16 AMP with GradScaler (no bf16 on 2060), L1 on log scale, seeds {0,1,2}.
**Baselines for the predictor:**
- constant = mean train log scale
- same head on DINOv2-ViT-S pooled features (optional, if time)
**Report:** median |log error| vs teacher, wall-clock training time, peak VRAM (`torch.cuda.max_memory_allocated`).
**Then:** E4 condition `learned` on Kaggle with X = predicted scale.

---

## 4. Main outputs for the paper

- **Table 1:** MRA per task × {baseline, placebo, shuffled, learned, teacher, LOO-rescale upper bound}, with 95% CIs.
- **Table 2:** decomposition: between-scene share per task, and distance < 2 m vs ≥ 2 m.
- **Figure 1:** elasticity plot: median log(pred/gt) vs log k per task, with the β = 1 reference line.
- **Figure 2 (optional):** method sketch: frozen VLM + one-line hint; predictor on frozen tokens.
- **Line for the abstract:** predictor size, training minutes on a 6 GB RTX 2060.

All numbers must come from `results/*.jsonl` via one script `analysis/make_tables.py`. No hand-typed numbers.

---

## 5. Repo layout

```
scale-hint/
  README.md            # exact commands, model ids, versions, deviations
  splits/scenes.json
  eval/run_vlm.py      # --condition {baseline,teacher,shuffled,placebo,k0.5,...,learned}
  eval/parse.py        # number extraction + unit handling (cm→m, m², etc.)
  teacher/run_mapanything.py
  predictor/cache_qwen_vis.py
  predictor/train.py
  analysis/decompose.py
  analysis/elasticity.py
  analysis/make_tables.py
  results/  cache/  figures/
```

`run_vlm.py` must append per question and resume from existing output (Kaggle sessions die). Log GPU, dtype, transformers version, frame count.

---

## 6. Timeline (start immediately)

| Block | Task | Where | Est. |
|---|---|---|---|
| 1 | Data download, split, `run_vlm.py`, E0 baseline | Kaggle | 2 h |
| 1 (parallel) | E3 teacher scale | 2060 | 2 h |
| 2 | E2 decomposition → **go/no-go** | CPU | 0.5 h |
| 3 | E4 hint runs (7 conditions) | Kaggle | 3 h |
| 3 (parallel) | E6 cache + train predictor | Kaggle + 2060 | 2 h |
| 4 | `learned` run, E5 elasticity, tables/figures | Kaggle + CPU | 1.5 h |
| 5 | Writing | — | 4–5 h |

If behind schedule, drop in this order: DINOv2 baseline → `teacher_ext` → k=0.71/1.41 → learned predictor. **Never drop E2 or E5**; they are the contribution.

---

## 7. Risks and checks

- **Anchoring:** the model copies the hint number. Detected by copy rate and elasticity; report it as a finding.
- **Negative text-hint result is weaker than positive:** the model may know scale but not use text numbers. State this limitation.
- **Label noise in VSI-Bench:** prefer ReVSI if available; report the < 2 m vs ≥ 2 m split.
- **Teacher error:** MapAnything scale may be off; the LOO-rescale bound does not depend on it.
- **Parsing:** log unparseable answers; count them as wrong (MRA 0), and report the rate per condition.
- **Leakage:** predictor never sees eval scenes; check the split file before training.

---

## 8. Writing skeleton (4 pages)

1. Intro: metric failures of VLMs; the four error sources; "we ask how much is scale".
2. Related work: tool pipelines, 3D fusion, scale tokens (all fine-tune or add tools; none decomposes).
3. Method: decomposition, hint protocol + controls, elasticity, tiny predictor.
4. Results: Table 1, Table 2, Figure 1.
5. Discussion: what to invest in; limitations (text channel, indoor-only, one VLM).
