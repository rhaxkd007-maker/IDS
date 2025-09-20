### 2.1 src/build_dataset.py — 7:1:2 split 확정, 설명 강화

# -*- coding: utf-8 -*-
"""
CIC-IDS-2017 전처리 + 데이터 분할(Train:Val:Test = 가변, 기본 0.7/0.1/0.2)
- 컬럼 정리, 라벨 표준화(WebAttack 교정, Begin 병합/분리),
- 수치 피처만 사용, ±inf→NaN→중앙값 대체, 전부 NaN/상수열 제거
- 결과 저장: X/y_{train,val,test}.parquet, feature_names.txt, label_encoder.json
- 리포트: class_distribution.json, feature_stats_by_label.csv, feature_schema.csv, X_sample_head.csv
"""
from __future__ import annotations
import re, glob, json, argparse
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def enforce_min_train_per_class(X_tr, y_tr, X_val, y_val, X_te, y_te, min_train=4):
    """각 클래스가 train에 최소 min_train개 이상 포함되도록
    val/test에서 부족분을 train으로 '끌어오는' 보정.
    - 데이터누수 방지: 끌려온 샘플은 해당 세트에서 제거됨.
    """
    import numpy as np, pandas as pd
    from collections import Counter

    X_tr = X_tr.copy(); X_val = X_val.copy(); X_te = X_te.copy()
    y_tr = np.asarray(y_tr, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    y_te = np.asarray(y_te, dtype=int)

    total = len(y_tr) + len(y_val) + len(y_te)
    tr_cnt = Counter(y_tr)

    def _pull_from(source_X, source_y, need_cls, need_num):
        idx = np.where(source_y == need_cls)[0][:need_num]
        if idx.size == 0:
            return None
        return idx

    for cls, cnt in list(tr_cnt.items()):
        if cnt >= min_train:
            continue
        need = min_train - cnt

        # 1) val에서 끌어오기
        idx = _pull_from(X_val, y_val, cls, need)
        if idx is not None:
            X_tr = pd.concat([X_tr, X_val.iloc[idx]], axis=0)
            y_tr = np.concatenate([y_tr, y_val[idx]])
            keep = np.ones(len(y_val), dtype=bool); keep[idx] = False
            X_val = X_val.iloc[keep]; y_val = y_val[keep]
            need -= len(idx)
            if need <= 0:
                continue

        # 2) test에서 끌어오기
        idx = _pull_from(X_te, y_te, cls, need)
        if idx is not None:
            X_tr = pd.concat([X_tr, X_te.iloc[idx]], axis=0)
            y_tr = np.concatenate([y_tr, y_te[idx]])
            keep = np.ones(len(y_te), dtype=bool); keep[idx] = False
            X_te = X_te.iloc[keep]; y_te = y_te[keep]

    assert len(y_tr) + len(y_val) + len(y_te) == total, "size mismatch after enforcement"
    return X_tr, y_tr, X_val, y_val, X_te, y_te
# ========================= 공통 유틸 =========================
CORE_STATS_FEATURES = [
    "Flow Duration","Flow Bytes/s","Flow Packets/s",
    "Packet Length Mean","Packet Length Std",
    "Fwd Packets/s","Bwd Packets/s","Total Fwd Packets","Total Backward Packets",
]

_DEF_OUTDIR = Path(__file__).resolve().parents[1] / "data/processed"
_DEF_REPORTS = Path(__file__).resolve().parents[1] / "reports"


def ensure_dirs(*paths: Path | str):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def clean_column_names(cols: List[str]) -> List[str]:
    out, seen = [], {}
    for c in cols:
        cc = ("" if c is None else str(c)).replace("\t", " ").strip()
        while "  " in cc:
            cc = cc.replace("  ", " ")
        cc = cc.replace(" ,", ",").replace(", ", ", ").lstrip()
        base = cc
        if base in seen:
            seen[base] += 1
            cc = f"{base}.{seen[base]}"  # 중복 컬럼명 방지
        else:
            seen[base] = 0
        out.append(cc)
    return out


_BASE_MAPPING = {
    "WEB ATTACK - BRUTE FORCE":"WebAttack-BruteForce",
    "WEB ATTACK - XSS":"WebAttack-XSS",
    "WEB ATTACK - SQL INJECTION":"WebAttack-SQLi",
    "WEB ATTACK - SQL INJECTION.":"WebAttack-SQLi",
    "WEB ATTACK - SQL":"WebAttack-SQLi",
    "DOS HULK":"DoS Hulk","DOS SLOWLORIS":"DoS slowloris","DOS SLOWHTTPTEST":"DoS Slowhttptest",
    "DDOS":"DDoS","PORTSCAN":"PortScan","BOT":"Bot","INFILTRATION":"Infiltration",
    "FTP-PATATOR":"FTP-Patator","SSH-PATATOR":"SSH-Patator","BENIGN":"BENIGN",
}
_BEGIN_RE = re.compile(r"(?:^|\b|[-_/()])\s*(begin|start|initial)\s*(?:$|\b)", re.I)

def _squash(x: str) -> str:
    import re
    x = x.replace("�","-").replace("–","-").replace("—","-")
    x = x.replace("_"," ")
    x = re.sub(r"\s*-\s*", " - ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def _canonical_webattack(xu: str) -> str | None:
    for k, v in _BASE_MAPPING.items():
        if k.startswith("WEB ATTACK") and xu == k:
            return v
    if xu.startswith("WEB ATTACK"):
        if   "BRUTE" in xu: return "WebAttack-BruteForce"
        elif "XSS"   in xu: return "WebAttack-XSS"
        elif "SQL"   in xu: return "WebAttack-SQLi"
        return "WebAttack-Other"
    return None

def _canonical_base(x: str) -> str:
    xu = x.upper()
    web = _canonical_webattack(xu)
    if web: return web
    if xu in _BASE_MAPPING: return _BASE_MAPPING[xu]
    xu = xu.replace("WEB ATTACK-","WEB ATTACK - ")
    if xu in _BASE_MAPPING: return _BASE_MAPPING[xu]
    return x

def _strip_begin(x: str) -> tuple[str,bool]:
    had = False
    if _BEGIN_RE.search(x):
        had = True
        x = _BEGIN_RE.sub("", x)
        x = _squash(x)
    return x, had

def normalize_label(s: str, phase_policy: str = "split") -> str:
    if pd.isna(s): return "UNKNOWN"
    x = _squash(str(s).strip())
    x0, had_begin = _strip_begin(x)
    base = _canonical_base(x0)
    if base.upper() == "BENIGN":
        return "BENIGN"
    return f"{base} - Begin" if (had_begin and phase_policy == "split") else base

# ------------------------- 로딩/정제 -------------------------

def read_csv_safely(p: str) -> pd.DataFrame:
    df = pd.read_csv(p, low_memory=False)
    df.columns = clean_column_names(df.columns.tolist())
    return df

def resolve_input_files(inputs: List[str] | None,
                        inputs_glob: List[str] | None,
                        inputs_dir: List[str] | None) -> List[str]:
    files: List[str] = []
    for p in (inputs or []):
        files.append(p)
    import glob
    for pat in (inputs_glob or []):
        files.extend(glob.glob(pat, recursive=True))
    from pathlib import Path
    for d in (inputs_dir or []):
        files.extend([str(p) for p in Path(d).rglob("*.csv")])
    if not files:
        # 자동탐색 (친절 모드)
        cands = []
        for d in [Path("data/raw"), Path("data"), Path.cwd()]:
            if d.exists():
                cands += [str(p) for p in d.rglob("*.csv")]
        files = cands
        if files:
            print(f"[WARN] 명시적 입력 없음 → 자동탐색 {len(files)}개 CSV")
        else:
            print("[INFO] 유효한 CSV가 없어 종료합니다.")
            return []
    uniq, seen = [], set()
    for f in files:
        f2 = str(Path(f))
        if f2.lower().endswith('.csv') and Path(f2).exists() and f2 not in seen:
            seen.add(f2); uniq.append(f2)
    return uniq

def load_and_concat(paths: List[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        print(f"[INFO] Loading: {p}")
        dfs.append(read_csv_safely(p))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"[INFO] Concatenated shape: {df.shape}")
    return df

def split_features_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    label_col = None
    for c in df.columns:
        if c.strip().lower() == "label":
            label_col = c; break
    if label_col is None:
        raise ValueError("Label 컬럼을 찾을 수 없습니다. (예: 'Label')")

    X = df.drop(columns=[label_col])
    y_raw = df[label_col]

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    X = X[numeric_cols].copy().replace([np.inf, -np.inf], np.nan)

    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        print(f"[INFO] Drop all-NaN cols: {len(all_nan)} … {all_nan[:8]}")
        X.drop(columns=all_nan, inplace=True)

    nunique = X.nunique(dropna=True)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f"[INFO] Drop constant cols: {len(const_cols)} … {const_cols[:8]}")
        X.drop(columns=const_cols, inplace=True)

    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X, y_raw

# ------------------------- 저장/리포트 -------------------------

def make_label_encoder(y: pd.Series) -> Dict[str,int]:
    classes = sorted(y.unique().tolist())
    return {c: i for i, c in enumerate(classes)}

def encode_labels(y: pd.Series, label2id: Dict[str,int]) -> np.ndarray:
    return y.map(label2id).astype(int).values

def save_basic_reports(y_std: pd.Series, X: pd.DataFrame, out_reports: Path):
    ensure_dirs(out_reports)
    (out_reports/"class_distribution.json").write_text(
        json.dumps(y_std.value_counts().sort_index().to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8")
    rows = []
    for label in sorted(y_std.unique().tolist()):
        sub = X[y_std == label]
        stat = {"Label": label, "Count": int(len(sub))}
        for col in CORE_STATS_FEATURES:
            if col in X.columns:
                s = sub[col]
                stat[f"{col}__mean"] = float(np.nanmean(s))
                stat[f"{col}__std"]  = float(np.nanstd(s))
                stat[f"{col}__p50"]  = float(np.nanpercentile(s, 50))
                stat[f"{col}__p90"]  = float(np.nanpercentile(s, 90))
            else:
                stat[f"{col}__mean"] = stat[f"{col}__std"] = stat[f"{col}__p50"] = stat[f"{col}__p90"] = None
        rows.append(stat)
    pd.DataFrame(rows).to_csv(out_reports/"feature_stats_by_label.csv", index=False, encoding="utf-8-sig")
    schema = pd.DataFrame({
        "feature": X.columns,
        "dtype": [str(X[c].dtype) for c in X.columns],
        "n_unique": [X[c].nunique(dropna=True) for c in X.columns],
        "n_missing": [int(X[c].isna().sum()) for c in X.columns],
    })
    schema.to_csv(out_reports/"feature_schema.csv", index=False, encoding="utf-8-sig")
    X.head(5).to_csv(out_reports/"X_sample_head.csv", index=False, encoding="utf-8-sig")

    # ------------------------- 메인 -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=None)
    ap.add_argument("--inputs-dir", nargs="+", default=None)
    ap.add_argument("--inputs-glob", nargs="+", default=None)
    ap.add_argument("--phase-policy", choices=["split","merge"], default="split")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio",   type=float, default=0.1)
    ap.add_argument("--test-ratio",  type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default=str(_DEF_OUTDIR))
    ap.add_argument("--reports", type=str, default=str(_DEF_REPORTS))
    ap.add_argument("--print-cols", type=int, default=40)
    args = ap.parse_args()

    outdir   = Path(args.outdir)
    reports  = Path(args.reports)
    ensure_dirs(outdir, reports)

    # 0) 입력 수집
    csv_files = resolve_input_files(args.inputs, args.inputs_glob, args.inputs_dir)
    if not csv_files:
        return

    # 1) 로드 & 결합
    df_all = load_and_concat(csv_files)

    # 2) 피처/라벨 분리 + 정제
    X, y_raw = split_features_labels(df_all)

    if args.print_cols > 0:
        preview = ", ".join(list(X.columns[:args.print_cols]))
        print(f"[INFO] columns kept: {len(X.columns)}")
        print(f"[INFO] first {min(args.print_cols, len(X.columns))} cols: {preview}")
        if len(X.columns) > args.print_cols:
            print(f"[INFO] ... and {len(X.columns) - args.print_cols} more")

    # 3) 라벨 표준화 + 인코딩
    y_std = y_raw.map(lambda s: normalize_label(s, phase_policy=args.phase_policy))
    label2id = make_label_encoder(y_std)
    y = encode_labels(y_std, label2id)

    # 4) 7:1:2 분할 (두 번의 stratified split)
    #   a) test 20% 분리
    X_trval, X_te, y_trval, y_te = train_test_split(
        X, y, test_size=args.test_ratio, random_state=args.seed, stratify=y)
    #   b) val 10%가 전체에서 되도록 → trval 대비 비율 = val/(train+val)
    val_rel = args.val_ratio / (args.train_ratio + args.val_ratio)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=val_rel, random_state=args.seed, stratify=y_trval)

    X_tr, y_tr, X_val, y_val, X_te, y_te = enforce_min_train_per_class(
    X_tr, y_tr, X_val, y_val, X_te, y_te, min_train=4
    )
    # 5) 저장 (parquet + 보조 파일)
    outdir.mkdir(parents=True, exist_ok=True)
    to_pq = lambda obj, path: (obj.to_parquet(path, index=False) if hasattr(obj, 'to_parquet') else pd.DataFrame({"label": obj}).to_parquet(path, index=False))
    to_pq(X_tr, outdir/"X_train.parquet"); to_pq(pd.DataFrame({"label": y_tr}), outdir/"y_train.parquet")
    to_pq(X_val, outdir/"X_val.parquet"); to_pq(pd.DataFrame({"label": y_val}), outdir/"y_val.parquet")
    to_pq(X_te, outdir/"X_test.parquet"); to_pq(pd.DataFrame({"label": y_te}), outdir/"y_test.parquet")

    # feature_names / label_encoder
    (outdir/"feature_names.txt").write_text("\n".join(X.columns), encoding="utf-8")
    (outdir/"label_encoder.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8")

    # 리포트
    save_basic_reports(pd.Series(y_std), X, reports)

    # 로그 요약
    print("[INFO] ===== Build Completed =====")
    print(f"[INFO] features={X.shape[1]} | classes={len(label2id)}")
    print(f"[INFO] split → train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")

if __name__ == "__main__":
    main()