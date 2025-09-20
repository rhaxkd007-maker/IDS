# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from .config import RAW_LABEL_COL, MAIN_LABEL_COL, SUFFIX_COL, DROP_IF_PRESENT, ID_COL
from .utils import normalize_columns, coerce_numeric, safe_fillna, split_begin_end, summarize_label_counts

def load_all_csvs(data_dir: Path) -> pd.DataFrame:
    paths = sorted([p for p in Path(data_dir).glob("*.csv")])
    if not paths:
        raise FileNotFoundError(f"No CSVs in {data_dir}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df = normalize_columns(df)
        if RAW_LABEL_COL not in df.columns:
            # try fallback common aliases
            for c in df.columns:
                if c.lower().strip()=="label":
                    df = df.rename(columns={c: RAW_LABEL_COL})
                    break
        df[MAIN_LABEL_COL], df[SUFFIX_COL] = zip(*df[RAW_LABEL_COL].map(split_begin_end))
        df.insert(0, ID_COL, np.arange(len(df)) + 1)
        dfs.append(df)
    full = pd.concat(dfs, axis=0, ignore_index=True)
    return full

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    # Remove exact duplicates on all feature columns (excluding IDs and label columns)
    drop_cols = set([ID_COL, RAW_LABEL_COL, MAIN_LABEL_COL, SUFFIX_COL]) | DROP_IF_PRESENT
    feature_cols = [c for c in df.columns if c not in drop_cols]
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols + [MAIN_LABEL_COL])  # preserve label-consistent duplicates
    after = len(df)
    df.attrs["dedup_removed"] = int(before - after)
    return df

def downsample_benign(df, factor=2, random_state=42):
    # factor: 공격 총합 대비 BENIGN 비율 (예: 2 => BENIGN = attack_sum * 2)
    label_col = "Label"  # 환경에 맞게 수정 필요
    attacks = df[~df[label_col].str.contains("BENIGN", case=False)]
    benign = df[df[label_col].str.contains("BENIGN", case=False)]
    target_benign = min(len(benign), int(len(attacks) * factor))
    if len(benign) <= target_benign:
        return df
    benign_down = benign.sample(n=target_benign, random_state=random_state)
    new_df = pd.concat([attacks, benign_down], axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return new_df

def select_features(df: pd.DataFrame):
    # Drop obvious non-feature columns if present
    drop_cols = [c for c in df.columns if c in DROP_IF_PRESENT]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

def train_valid_test_split(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    # stratify by main label
    y = df[MAIN_LABEL_COL]
    df_train, df_test = train_test_split(df, test_size=test_size, stratify=y, random_state=random_state)
    y_train = df_train[MAIN_LABEL_COL]
    val_ratio = val_size / (1.0 - test_size)
    df_train, df_val = train_test_split(df_train, test_size=val_ratio, stratify=y_train, random_state=random_state)
    return df_train, df_val, df_test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size", type=float, default=0.1)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    (out_dir).mkdir(parents=True, exist_ok=True)

    df = load_all_csvs(data_dir)
    df = select_features(df)
    df = coerce_numeric(df, exclude_cols={RAW_LABEL_COL, MAIN_LABEL_COL, SUFFIX_COL, ID_COL})
    df = safe_fillna(df)
    df = deduplicate(df)
    df = downsample_benign(df, factor=2)
    
    # Save dedup stats
    meta = {
        "rows_after_dedup": len(df),
        "dedup_removed": int(df.attrs.get("dedup_removed", 0)),
        "label_main_summary": summarize_label_counts(df[MAIN_LABEL_COL]),
    }

    # Split
    tr, va, te = train_valid_test_split(df, args.test_size, args.val_size, args.random_state)

    # Persist
    for name, part in [("train", tr), ("val", va), ("test", te)]:
        part.to_parquet(out_dir / f"{name}.parquet", index=False)

    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[data_prep] done. Saved to", out_dir)

if __name__ == "__main__":
    main()
