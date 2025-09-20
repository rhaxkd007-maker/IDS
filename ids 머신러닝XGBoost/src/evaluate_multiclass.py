# src/evaluate_multiclass.py
# -*- coding: utf-8 -*-
"""
멀티클래스 최종 평가 (Test 전용)
- 입력: data/processed/{X_test.parquet, y_test.parquet, feature_names.txt, label_encoder.json}
- 모델: models/xgb_multiclass.json
- 산출물(reports/):
    test_metrics.json
    classification_report.txt
    per_class_report.csv                (precision/recall/f1/support + TP/FP/FN/TN + accuracy_ova)
    confusion_matrix.csv                (정수)
    confusion_matrix_norm.csv           (row-normalized)
    confusion_matrix_test_raw.png       (테스트셋, 원본)
    confusion_matrix_test_norm.png      (테스트셋, 정규화)
    test_predictions.csv                (정답/예측/Top-3)
    feature_importance_all.csv          (gain/weight/cover 결합)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ──────────────────────────────────────────────────────────────────────────────
# 설정/유틸
# ──────────────────────────────────────────────────────────────────────────────

NON_FEATURE_TOKENS = {
    "precision", "recall", "f1", "f1-score", "support",
    "accuracy", "macro avg", "weighted avg",
}

def _log(msg: str) -> None:
    print(msg, flush=True)

def clean_feature_list(raw_list: List[str]) -> List[str]:
    """feature_names.txt에서 지표/쓰레기 토큰 제거 + 중복 제거"""
    keep, seen = [], set()
    for c in raw_list:
        c = str(c).strip()
        if not c or c.lower() in NON_FEATURE_TOKENS:
            continue
        if c not in seen:
            keep.append(c)
            seen.add(c)
    return keep

def align_to_model_features(X: pd.DataFrame, model_feats: List[str]) -> pd.DataFrame:
    """
    모델이 기억하는 feature_names 순서에 맞춤:
      - 누락 피처는 0.0으로 추가
      - 모델에 없는 피처는 드롭
      - 순서 정확히 맞춰 반환
    """
    X = X.copy()
    exist = set(X.columns)
    add = [c for c in model_feats if c not in exist]
    for c in add:
        X[c] = 0.0
    drop = [c for c in X.columns if c not in model_feats]
    if drop:
        X.drop(columns=drop, inplace=True, errors="ignore")
    X = X[model_feats]
    if add:
        _log(f"[WARN] 테스트셋에 없던 피처 {len(add)}개를 0.0으로 추가: {add[:10]}{' ...' if len(add)>10 else ''}")
    if drop:
        _log(f"[WARN] 모델에 없는 테스트 피처 {len(drop)}개 드롭: {drop[:10]}{' ...' if len(drop)>10 else ''}")
    return X

# ──────────────────────────────────────────────────────────────────────────────
# 로딩
# ──────────────────────────────────────────────────────────────────────────────

def load_processed_test(data_dir: Path) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, str], List[str]]:
    """X_test/y_test + feature_names.txt + label_encoder.json 로드"""
    X_te = pd.read_parquet(data_dir / "X_test.parquet")
    y_te = pd.read_parquet(data_dir / "y_test.parquet")["label"].astype(int).values

    # object → number 안전 변환
    for c in X_te.columns:
        if X_te[c].dtype == "object":
            X_te[c] = pd.to_numeric(X_te[c], errors="coerce")

    feat_txt = data_dir / "feature_names.txt"
    feat_list = clean_feature_list(feat_txt.read_text(encoding="utf-8").splitlines()) if feat_txt.exists() else list(X_te.columns)

    # 라벨 매핑(원라벨 유지)
    enc_path = data_dir / "label_encoder.json"
    if enc_path.exists():
        enc = json.loads(enc_path.read_text(encoding="utf-8"))
        mapping = enc.get("mapping", enc)          # 두 형태 모두 지원
        id2lab = {int(v): str(k) for k, v in mapping.items()}
    else:
        n_classes = int(np.max(y_te) + 1)
        id2lab = {i: str(i) for i in range(n_classes)}

    return X_te, y_te, id2lab, feat_list

def load_booster(model_path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    fns = getattr(booster, "feature_names", None)
    _log(f"[INFO] booster loaded. feature_names={len(fns) if fns else 0}")
    return booster

# ──────────────────────────────────────────────────────────────────────────────
# 예측 (버전 호환)
# ──────────────────────────────────────────────────────────────────────────────

def predict_proba_compat(booster: xgb.Booster, X_df: pd.DataFrame, n_classes: int) -> np.ndarray:
    """
    1) numpy float32로 inplace_predict
    2) 실패 시 DMatrix로 predict 폴백
    3) shape 검증/보정
    """
    if X_df is None or X_df.shape[0] == 0 or X_df.shape[1] == 0:
        raise ValueError(f"테스트 입력이 비었습니다. X shape={None if X_df is None else X_df.shape}")

    X_np = X_df.to_numpy(dtype=np.float32, copy=False)

    # 경로 1: inplace_predict
    proba = None
    try:
        proba = booster.inplace_predict(X_np, validate_features=False)
    except Exception as e1:
        _log(f"[WARN] inplace_predict 실패 → {type(e1).__name__}: {e1}")

    # 경로 2: DMatrix 예측
    if proba is None:
        try:
            dmat = xgb.DMatrix(X_np, feature_names=list(X_df.columns))
            proba = booster.predict(dmat)
        except Exception as e2:
            raise RuntimeError(
                f"DMatrix.predict 실패: {type(e2).__name__}: {e2}\n"
                f"  X_np.shape={X_np.shape}, n_classes={n_classes}"
            ) from e2

    proba = np.asarray(proba)
    if proba.ndim == 1:
        # (n_samples * n_classes,) → (n_samples, n_classes)
        if proba.size % X_np.shape[0] != 0:
            raise RuntimeError("1D 예측 결과 크기 불일치")
        proba = proba.reshape(X_np.shape[0], -1)
    if proba.shape[1] != n_classes:
        try:
            proba = proba.reshape(X_np.shape[0], n_classes)
            _log(f"[WARN] 예측 차원을 n_classes={n_classes}로 보정")
        except Exception:
            raise RuntimeError(f"예측 shape 불일치: {proba.shape} vs (*,{n_classes})")
    return proba

# ──────────────────────────────────────────────────────────────────────────────
# 메트릭 저장 + 시각화
# ──────────────────────────────────────────────────────────────────────────────

def save_confusion_matrices(rep_dir: Path, cm: np.ndarray, labels: List[str]) -> None:
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(rep_dir / "confusion_matrix.csv", encoding="utf-8-sig")
    with np.errstate(all="ignore"):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    pd.DataFrame(cm_norm, index=labels, columns=labels).to_csv(rep_dir / "confusion_matrix_norm.csv", encoding="utf-8-sig")
    _log("[INFO] Saved confusion_matrix*.csv")

def save_confusion_png(rep_dir: Path, cm: np.ndarray, labels: List[str]) -> None:
    """테스트셋 혼동행렬 PNG(raw & normalized) 저장"""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _draw(m: np.ndarray, title: str, out: Path):
        plt.figure(figsize=(max(6, len(labels)*0.55), max(5, len(labels)*0.55)))
        im = plt.imshow(m, interpolation="nearest")
        plt.title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        t = np.arange(len(labels))
        plt.xticks(t, labels, rotation=90)
        plt.yticks(t, labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()

    _draw(cm, "Confusion Matrix (test, raw)",   rep_dir / "confusion_matrix_test_raw.png")
    with np.errstate(all="ignore"):
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cmn = np.nan_to_num(cmn)
    _draw(cmn, "Confusion Matrix (test, normalized)", rep_dir / "confusion_matrix_test_norm.png")

def save_per_class_report(
    rep_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    labels: List[str],
) -> None:
    """classification_report + OVA accuracy + TP/FP/FN/TN 저장"""
    # 텍스트 리포트
    txt = classification_report(y_true, y_pred, target_names=labels, digits=4, zero_division=0)
    (rep_dir / "classification_report.txt").write_text(txt, encoding="utf-8")
    (rep_dir / "classification_report_test.txt").write_text(txt, encoding="utf-8")
    # 딕셔너리 리포트 + OVA accuracy
    rep = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    total = int(cm.sum())
    rows = []
    for i, lab in enumerate(labels):
        TP = int(cm[i, i])
        FN = int(cm[i, :].sum() - TP)
        FP = int(cm[:, i].sum() - TP)
        TN = int(total - TP - FN - FP)
        acc_ova = float((TP + TN) / total) if total else 0.0

        d = rep.get(lab, {"precision": 0, "recall": 0, "f1-score": 0, "support": 0})
        rows.append({
            "label": lab,
            "precision": float(d["precision"]),
            "recall": float(d["recall"]),
            "f1": float(d["f1-score"]),
            "support": int(d["support"]),
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "accuracy_ova": acc_ova,
        })
    pd.DataFrame(rows).to_csv(rep_dir / "per_class_report.csv", index=False, encoding="utf-8-sig")
    _log("[INFO] Saved per_class_report.csv")

def save_predictions(rep_dir: Path, proba: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, id2lab: Dict[int, str]) -> None:
    """샘플별 예측/Top-3 저장"""
    top1 = np.max(proba, axis=1)
    top3_idx = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
    top3_labels = [[id2lab.get(j, str(j)) for j in row] for row in top3_idx]
    top3_scores = [[float(proba[i, j]) for j in row] for i, row in enumerate(top3_idx)]

    df = pd.DataFrame({
        "true_id": y_true,
        "pred_id": y_pred,
        "true_label": [id2lab.get(i, str(i)) for i in y_true],
        "pred_label": [id2lab.get(i, str(i)) for i in y_pred],
        "proba_top1": top1,
    })
    for k in range(3):
        df[f"top{k+1}_label"] = [row[k] if len(row) > k else "" for row in top3_labels]
        df[f"top{k+1}_proba"] = [row[k] if len(row) > k else np.nan for row in top3_scores]
    df.to_csv(rep_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    _log("[INFO] Saved test_predictions.csv")

def save_feature_importance(rep_dir: Path, booster: xgb.Booster, X_cols: List[str]) -> None:
    """gain/weight/cover 결합 후 CSV로 저장 (f0→원컬럼명 매핑 시도)"""
    try:
        score_gain   = booster.get_score(importance_type="gain")
        score_weight = booster.get_score(importance_type="weight")
        score_cover  = booster.get_score(importance_type="cover")

        imp_gain   = pd.DataFrame({"feature": list(score_gain.keys()),   "gain":   list(score_gain.values())})
        imp_weight = pd.DataFrame({"feature": list(score_weight.keys()), "weight": list(score_weight.values())})
        imp_cover  = pd.DataFrame({"feature": list(score_cover.keys()),  "cover":  list(score_cover.values())})

        imp = imp_gain.merge(imp_weight, on="feature", how="outer").merge(imp_cover, on="feature", how="outer").fillna(0.0)
        mapping = {f"f{i}": col for i, col in enumerate(X_cols)}  # f0→원래 컬럼명
        imp["feature"] = imp["feature"].map(lambda k: mapping.get(k, k))
        imp.sort_values("gain", ascending=False, inplace=True)
        imp.to_csv(rep_dir / "feature_importance_all.csv", index=False, encoding="utf-8-sig")
        _log("[INFO] Saved feature_importance_all.csv")
    except Exception as e:
        _log(f"[WARN] feature importance 저장 스킵: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   type=str, default="data/processed")
    p.add_argument("--models_dir", type=str, default="models")
    p.add_argument("--reports_dir",type=str, default="reports")
    p.add_argument("--topk",       type=int, default=3, help="Top-K accuracy의 K")
    args = p.parse_args()

    data_dir  = Path(args.data_dir)
    models_dir= Path(args.models_dir)
    reports   = Path(args.reports_dir); reports.mkdir(parents=True, exist_ok=True)

    # 1) 데이터 로드
    X_te, y_te, id2lab, feat_list = load_processed_test(data_dir)

    # 2) 모델 로드
    model_path = models_dir / "xgb_multiclass.json"
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일 없음: {model_path}")
    booster = load_booster(model_path)

    # 3) 피처 정합성 맞추기 (모델 기준)
    model_feats = list(getattr(booster, "feature_names", None) or feat_list)
    X_te = align_to_model_features(X_te, model_feats)
    n_classes = len(id2lab)

    # 4) 예측
    proba  = predict_proba_compat(booster, X_te, n_classes)
    y_pred = np.argmax(proba, axis=1)

    # 5) 전체 지표(+ Top-K)
    k = max(1, int(args.topk))
    topk_idx = np.argsort(proba, axis=1)[:, -k:]
    topk_acc = float(np.mean([y_te[i] in topk_idx[i] for i in range(len(y_te))]))
    metrics = {
        "test_accuracy": float(accuracy_score(y_te, y_pred)),
        "test_f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "test_balanced_acc": float(balanced_accuracy_score(y_te, y_pred)),
        f"test_top{k}_acc": topk_acc,
        "n_classes": n_classes,
        "n_test": int(len(y_te)),
    }
    (reports / "test_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"[INFO] test_acc={metrics['test_accuracy']:.4f} | macroF1={metrics['test_f1_macro']:.4f} "
         f"| bal_acc={metrics['test_balanced_acc']:.4f} | top{k}_acc={topk_acc:.4f}")

    # 6) 혼동행렬 + per-class 리포트 + PNG
    label_names = [id2lab.get(i, str(i)) for i in range(n_classes)]
    cm = confusion_matrix(y_te, y_pred, labels=list(range(n_classes)))
    save_confusion_matrices(reports, cm, label_names)
    save_confusion_png(reports, cm, label_names)
    save_per_class_report(reports, y_te, y_pred, cm, label_names)

    # 7) 예측 상세 & 중요도
    save_predictions(reports, proba, y_te, y_pred, id2lab)
    save_feature_importance(reports, booster, list(X_te.columns))

    _log("[INFO] ===== Test Evaluation Completed =====")

if __name__ == "__main__":
    main()