import json
import numpy as np
import pandas as pd
from pathlib import Path

def save_text_report(path: Path, lines):
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(str(ln) + "\n")

def aggregate_metrics(per_class_df: pd.DataFrame, overall: dict) -> dict:
    # overall already includes accuracy, macroF1, balanced_accuracy etc.
    # We'll add weighted averages from per-class
    w = per_class_df["support"].values
    w = w / w.sum()
    wa_prec = float(np.sum(w * per_class_df["precision"].values))
    wa_rec = float(np.sum(w * per_class_df["recall"].values))
    wa_f1 = float(np.sum(w * per_class_df["f1"].values))
    agg = dict(overall)
    agg.update({
        "weighted_precision": wa_prec,
        "weighted_recall": wa_rec,
        "weighted_f1": wa_f1
    })
    return agg
