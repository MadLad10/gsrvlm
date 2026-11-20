#!/usr/bin/env bash
# MapAnything for run_teacher.py. Run this in its OWN session (Colab or a second
# Kaggle notebook): it can change package versions the VLM side depends on.
set -euo pipefail
python - <<'EOF'
import torch, torchaudio
print("torch", torch.__version__, "| torchaudio", torchaudio.__version__)
EOF
# MapAnything -> uniception requires torchaudio unpinned. Kaggle/Colab already ship
# a torchaudio matched to their torch, so pip leaves it alone. If the import check
# below fails with an ABI error, reinstall torchaudio at the torch version printed above.
if [ ! -d third_party/map-anything ]; then
  mkdir -p third_party
  git clone --depth 1 https://github.com/facebookresearch/map-anything.git third_party/map-anything
fi
pip install -q -e third_party/map-anything
python -c "import torch, torchaudio, mapanything; print('mapanything ok; torch', torch.__version__)"
