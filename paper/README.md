# Paper: "Compressed, Not Mis-Scaled"

AAAI-27 template (GSR-VLM workshop, long paper). `paper.tex` + `refs.bib`; compiled `paper.pdf`.
Every number in the paper comes from the files below; nothing is typed by hand except what is
copied from `numbers.json` / `results/tables.md`.

## Where each number comes from

| Paper content | Source file | Produced by |
|---|---|---|
| Qwen3-VL-4B runs (all conditions) | `results/<cond>.s0.jsonl` | `python run_vlm.py --condition <cond> [--split all]` |
| Qwen3-VL-8B runs | `results_8b/<cond>.s0.jsonl` | `python run_vlm.py --model Qwen/Qwen3-VL-8B-Instruct --out_dir results_8b --condition <cond>` (baseline with `--split all`) |
| Answer positive control (`answer`, `answer_k0.5`, `answer_k2`) | `results/answer*.s0.jsonl`, `results_8b/answer*.s0.jsonl` | condition `answer[_k]` in `scalehint/prompts.py` / `run_vlm.py`: prompt line "Hint: the correct answer is approximately X <unit>." with X = ground truth × k |
| Compression slope, reverse bound, ICC raw/residual, LOO rescale (Table 1) | `numbers.json` → `<model>.decomp` | `python paper/analysis.py` (reuses `decompose.analyse`) |
| Variance shares bias / scene / question (30/9/61 ...) | `numbers.json` → `<model>.budget` | `analysis.py: error_budget` (R² of e on log gt; ICC(1) of residuals × residual variance; remainder) |
| Quintile ratios (1.30→0.65, 1.50→0.63), top-5 answer shares (54%, 62%), log-log correlation | `numbers.json` → `<model>.compression` | `analysis.py: compression_bins` |
| Paired ΔMRA vs baseline with CIs (Table 2, appendix table) | `numbers.json` → `<model>.delta_k1`; `appendix_rows.tex` | `analysis.py: paired_delta`, `appendix_table` |
| Mean β with CI, median β, % up / down between k=0.5 and k=2 (Table 3) | `numbers.json` → `<model>.elasticity` | `analysis.py: elasticity` |
| Share of answers changed vs baseline (36–57%, placebo 26–39%) | `numbers.json` → `<model>.changed` | `analysis.py: change_rate` |
| Headroom / calibration / oracle anchor (Table 4) | `numbers.json` → `4b.headroom` | `analysis.py: headroom` (log-linear fits on train scenes, MRA on eval scenes) |
| Reasoning-prompt loop sensitivity | `results/spatialprompt_sensitivity.json` | `results_archive/spatialprompt_v1_512tok/spatialprompt_sensitivity.py` |
| Teacher sanity, predictor numbers | `results/tables.md` | `python make_tables.py --n_boot 1000` |
| Figures 2 and 5 | `fig_compression.pdf`, `fig_elasticity.pdf` | `analysis.py: figures` |
| Figures 1, 3, 6 (qualitative examples) | `fig_teaser.pdf`, `fig_qual_compression.pdf`, `fig_qual_random.pdf`; selected IDs in `teaser_examples.json`, `gallery_examples.json` | `paper/make_teaser.py`, `paper/make_gallery.py` (fixed selection rules, see docstrings) |

`print_numbers.py` prints every quoted value from `numbers.json` in the format used in the text.

## Reproduce

```bash
python paper/analysis.py --n_boot 1000   # numbers.json, appendix_rows.tex, fig_compression/elasticity
python paper/make_teaser.py              # fig_teaser.pdf (needs cache/frames from prepare_data.py)
python paper/make_gallery.py             # fig_qual_*.pdf
python paper/print_numbers.py
cd paper && pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

All CIs: 95% percentile intervals, 1000 scene-level bootstrap resamples, seed 0 (`analysis.py`).
Figures use matplotlib with `text.usetex` (needs a LaTeX install) so fonts embed as Type 1.

## Run environment

Local run on one RTX 5090, bf16 (not the T4/fp16 Kaggle setup); see the per-condition
`*.meta.json` files for model, dtype, torch/transformers versions.
