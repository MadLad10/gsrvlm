import glob
import os

import numpy as np
import pandas as pd

from . import ROOT_DEFAULTS
from .data import read_jsonl


def load(cond, results_dir=ROOT_DEFAULTS["results"]):
    rows = []
    for p in sorted(glob.glob(os.path.join(results_dir, f"{cond}.s*.jsonl"))):
        rows += read_jsonl(p)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("qid", keep="last").reset_index(drop=True)
    df["pred"] = pd.to_numeric(df["pred"], errors="coerce")
    df["ok"] = df["pred"].notna() & (df["pred"] > 0)
    df["e"] = np.where(df["ok"], np.log(df["pred"].where(df["ok"], 1.0) / df["gt"]), np.nan)
    return df


def conditions(results_dir=ROOT_DEFAULTS["results"]):
    names = set()
    for p in glob.glob(os.path.join(results_dir, "*.s*.jsonl")):
        names.add(os.path.basename(p).rsplit(".s", 1)[0])
    return sorted(names)
