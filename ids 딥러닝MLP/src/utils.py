import re
import numpy as np
import pandas as pd
from pathlib import Path
from .config import RAW_LABEL_COL, MAIN_LABEL_COL, SUFFIX_COL

def split_begin_end(label: str):
    # Split "...-Begin" or "...-End" or "...-Flow" variants, keep main and suffix
    if not isinstance(label, str):
        return label, ""
    m = re.match(r"^(.*?)-(Begin|End|Flow)$", label.strip())
    if m:
        return m.group(1).strip(), m.group(2)
    return label.strip(), ""

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (pd.Index(df.columns)
                  .astype(str)
                  .str.replace(r"\s+", " ", regex=True)
                  .str.strip())
    return df

def coerce_numeric(df: pd.DataFrame, exclude_cols):
    for c in df.columns:
        if c in exclude_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def safe_fillna(df: pd.DataFrame):
    # Replace inf/-inf and NaN with column medians (robust to outliers)
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            med = df[c].median()
            df[c] = df[c].fillna(med if np.isfinite(med) else 0.0)
    return df

def summarize_label_counts(y_series: pd.Series) -> dict:
    counts = y_series.value_counts(dropna=False).to_dict()
    total = int(y_series.shape[0])
    pct = {k: (v/total) for k,v in counts.items()}
    return {"counts": counts, "proportions": pct, "total": total}
