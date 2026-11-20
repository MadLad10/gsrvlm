#!/usr/bin/env bash
# Run one or more conditions, one process per visible GPU (2x T4 on Kaggle -> 2 shards).
#   bash run_conds.sh baseline                       # eval split
#   SPLIT=all bash run_conds.sh baseline             # all scenes (needed for E2)
#   EXTRA="--limit_per_task 100" bash run_conds.sh placebo
# Logs: logs/<condition>.s<k>.log
set -uo pipefail
SPLIT="${SPLIT:-eval}"
EXTRA="${EXTRA:-}"
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -lt 1 ] && { echo "no GPU visible"; exit 1; }
mkdir -p logs
for COND in "$@"; do
  echo "== $COND on $NGPU GPU(s), split=$SPLIT"
  pids=()
  for ((g=0; g<NGPU; g++)); do
    CUDA_VISIBLE_DEVICES=$g python run_vlm.py --condition "$COND" --split "$SPLIT" \
      --shard $g --nshards $NGPU $EXTRA > "logs/$COND.s$g.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  tail -n 3 logs/$COND.s*.log
  [ $fail -ne 0 ] && echo "!! $COND: a shard failed, see logs/$COND.s*.log (rerun resumes)"
done
