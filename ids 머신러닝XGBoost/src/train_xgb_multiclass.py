# src/train_xgb_multiclass.py
# -*- coding: utf-8 -*-
"""
멀티클래스 학습 (XGBoost)
- 입력: data/processed/{X_train,y_train,X_val,y_val}.parquet
- 기능:
  • 조기종료(기본 150, XGB 1.x/2.x 자동 호환)
  • GPU: tree_method="hist", device="cuda" (실패 시 CPU로 폴백)
  • 샘플가중치: none | inv | invsqrt
  • 혼동행렬: 원라벨명으로 CSV + PNG(raw/normalized) 저장
- 산출물:
  • models/xgb_multiclass.json
  • reports/train_metrics.json, classification_report.txt,
    confusion_matrix*.{csv,png}, feature_importance_*.csv, train_logloss.png
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)

# XGB 2.x 콜백(없으면 자동 우회)
try:
    from xgboost.callback import EarlyStopping as XgbEarlyStopping
except Exception:
    XgbEarlyStopping = None

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# 경로 기본값
# ──────────────────────────────────────────────────────────────────────────────
_DEF_DATA    = Path(__file__).resolve().parents[1] / "data/processed"
_DEF_MODELS  = Path(__file__).resolve().parents[1] / "models"
_DEF_REPORTS = Path(__file__).resolve().parents[1] / "reports"


# ──────────────────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────────────────
def _load_xy(data_dir: Path, split: str):
    X = pd.read_parquet(data_dir / f"X_{split}.parquet")
    y = pd.read_parquet(data_dir / f"y_{split}.parquet")["label"].astype(int).values
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, y

def _weights(y: np.ndarray, mode: str):
    if mode == "none":
        return None
    vals, cnts = np.unique(y, return_counts=True)
    freq = {v: c for v, c in zip(vals, cnts)}
    if mode == "inv":
        w = np.array([1.0 / freq[v] for v in y], dtype=float)
    elif mode == "invsqrt":
        w = np.array([1.0 / np.sqrt(freq[v]) for v in y], dtype=float)
    else:
        return None
    w *= (len(w) / w.sum())
    return w

def _load_id2lab(data_dir: Path):
    p = data_dir / "label_encoder.json"
    if not p.exists():
        return None
    enc = json.loads(p.read_text(encoding="utf-8"))
    mapping = enc.get("mapping", enc)  # 두 포맷 지원
    return {int(v): str(k) for k, v in mapping.items()}

def _cm_png(cm: np.ndarray, labels, out_png: Path, title="Confusion Matrix (val)", norm=False):
    plt.figure(figsize=(max(6, len(labels)*0.55), max(5, len(labels)*0.55)))
    m = cm.astype(float)
    if norm:
        with np.errstate(all="ignore"):
            m = m / m.sum(axis=1, keepdims=True)
        m = np.nan_to_num(m)
        title = f"{title} - normalized"
    im = plt.imshow(m, interpolation="nearest")  # 기본 colormap
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=90)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _best_iter(booster: xgb.Booster) -> int:
    try:
        a = booster.attributes()
        if "best_iteration" in a:
            return int(a["best_iteration"])
    except Exception:
        pass
    return int(getattr(booster, "best_iteration", -1))

def _fit(model, X_tr, y_tr, X_val, y_val, w_tr, rounds=150, verbose=50):
    # XGB 2.x 콜백 경로
    if XgbEarlyStopping is not None:
        try:
            return model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_val, y_val)],
                verbose=verbose,
                callbacks=[XgbEarlyStopping(rounds=rounds, save_best=True, maximize=False)],
            )
        except TypeError:
            pass
    # XGB 1.x 경로
    try:
        return model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=[(X_val, y_val)],
            verbose=verbose,
            early_stopping_rounds=rounds,
        )
    except TypeError:
        print("[WARN] early stopping 미지원 → 조기종료 없이 학습")
        return model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_val, y_val)], verbose=verbose)


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   type=str, default=str(_DEF_DATA))
    ap.add_argument("--models_dir", type=str, default=str(_DEF_MODELS))
    ap.add_argument("--reports_dir",type=str, default=str(_DEF_REPORTS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu",  action="store_true")
    # 하이퍼파라미터(실전 기본)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--n_estimators", type=int, default=800)
    ap.add_argument("--eta", type=float, default=0.08)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample_bytree", type=float, default=0.8)
    ap.add_argument("--min_child_weight", type=float, default=1.0)
    ap.add_argument("--reg_lambda", type=float, default=1.0)
    ap.add_argument("--reg_alpha",  type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=150)  # ← 기본 150
    ap.add_argument("--class_weight", choices=["none","inv","invsqrt"], default="invsqrt")
    args = ap.parse_args()

    data_dir   = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    reports    = Path(args.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드
    X_tr, y_tr = _load_xy(data_dir, "train")
    X_val, y_val = _load_xy(data_dir, "val")

    # 2) 샘플 가중치
    w_tr = _weights(y_tr, args.class_weight)
    if w_tr is not None:
        print(f"[INFO] Using sample weight: {args.class_weight}")

    # 3) 파라미터 구성 (XGB 2.x 권장: hist + device)
    n_classes = int(np.max(np.concatenate([y_tr, y_val])) + 1)
    params = {
        "objective": "multi:softprob",
        "num_class": n_classes,
        "max_depth": args.max_depth,
        "eta": args.eta,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "reg_alpha": args.reg_alpha,
        "eval_metric": "mlogloss",
        "verbosity": 1,
        "random_state": args.seed,
        "tree_method": "hist",
        "device": "cuda" if args.gpu else "cpu",
    }
    model = xgb.XGBClassifier(**params, n_estimators=args.n_estimators)
    print(f"[INFO] Training with device={params['device']}, tree_method={params['tree_method']}")

    # 4) 학습 (GPU 실패 시 CPU 폴백)
    try:
        _fit(model, X_tr, y_tr, X_val, y_val, w_tr, rounds=args.early_stopping_rounds, verbose=50)
    except xgb.core.XGBoostError:
        if args.gpu:
            print("[WARN] GPU 실패 → CPU(hist)로 폴백")
            params["device"] = "cpu"
            model = xgb.XGBClassifier(**params, n_estimators=args.n_estimators)
            _fit(model, X_tr, y_tr, X_val, y_val, w_tr, rounds=args.early_stopping_rounds, verbose=50)
        else:
            raise

    # 5) 검증 성능 요약
    y_pred = model.predict(X_val)
    acc  = float(accuracy_score(y_val, y_pred))
    f1m  = float(f1_score(y_val, y_pred, average="macro"))
    bacc = float(balanced_accuracy_score(y_val, y_pred))
    booster = model.get_booster()
    best_iter = _best_iter(booster)

    (reports / "train_metrics.json").write_text(
        json.dumps(
            {"val_accuracy": acc, "val_f1_macro": f1m, "val_balanced_acc": bacc, "best_iteration": best_iter},
            indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    # 6) 라벨명 복원 → 혼동행렬 저장 (CSV + PNG raw/norm)
    id2lab = _load_id2lab(data_dir) or {i: f"C{i}" for i in range(n_classes)}
    labels = [id2lab.get(i, f"C{i}") for i in range(n_classes)]
    cm = confusion_matrix(y_val, y_pred, labels=list(range(n_classes)))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(reports / "confusion_matrix.csv", encoding="utf-8-sig")
    _cm_png(cm, labels, reports / "confusion_matrix_raw.png",  title="Confusion Matrix (val)", norm=False)
    _cm_png(cm, labels, reports / "confusion_matrix_norm.png", title="Confusion Matrix (val)", norm=True)

    # 7) 분류 리포트/중요도/로그
    (reports / "classification_report_val.txt").write_text(
    classification_report(y_val, y_pred, target_names=labels, digits=4), encoding="utf-8"
    )

    for t in ["gain", "weight", "cover"]:
        sc = booster.get_score(importance_type=t)
        pd.DataFrame({"feature": list(sc.keys()), t: list(sc.values())}).sort_values(t, ascending=False)\
          .to_csv(reports / f"feature_importance_{t}.csv", index=False, encoding="utf-8-sig")

    # 학습 곡선 (있으면)
    try:
        ev = model.evals_result_
        if ev and "validation_0" in ev:
            metric = list(ev["validation_0"].keys())[0]
            vals = ev["validation_0"][metric]
            plt.figure()
            plt.plot(range(1, len(vals)+1), vals)
            plt.xlabel("Iteration")
            plt.ylabel(metric)
            plt.title("Eval metric over iterations")
            plt.tight_layout()
            plt.savefig(reports / "train_logloss.png", dpi=160)
            plt.close()
    except Exception:
        pass

    # 8) 모델 저장
    booster.save_model(str(models_dir / "xgb_multiclass.json"))
    (models_dir / "xgb_multiclass.best_iteration.txt").write_text(str(best_iter), encoding="utf-8")

    print("[INFO] ===== Training Completed =====")

if __name__ == "__main__":
    main()