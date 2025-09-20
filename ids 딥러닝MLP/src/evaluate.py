# src/evaluate.py
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
)

from tensorflow.keras.models import load_model

# ===== 유틸 =====
MAIN_LABEL_COL = "LabelMain"
SUFFIX_COL = "LabelSuffix"
ID_COL = "RowID"
RAW_LABEL_COL = "Label"

def split_Xy(df: pd.DataFrame):
    X = df.drop(columns=[MAIN_LABEL_COL, SUFFIX_COL, ID_COL, RAW_LABEL_COL], errors="ignore")
    y = df[MAIN_LABEL_COL].astype(str)
    return X, y

def save_cm(cm, labels, out_png_raw, out_png_norm):
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=False, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (counts)")
    plt.tight_layout(); plt.savefig(out_png_raw); plt.close()

    cmn = (cm.T / cm.sum(axis=1)).T
    plt.figure(figsize=(8,6))
    sns.heatmap(cmn, annot=False, xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix (row-normalized)")
    plt.tight_layout(); plt.savefig(out_png_norm); plt.close()
    return cm, cmn

# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)  # out/data (parquet들 위치)
    ap.add_argument("--out_dir", required=True)   # out (models, reports 위치)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 데이터 로드 (parquet)
    te = pd.read_parquet(data_dir / "test.parquet")
    va = pd.read_parquet(data_dir / "val.parquet")

    # 스케일러/라벨인코더/모델 로드
    scaler = joblib.load(out_dir / "models" / "scaler.joblib")
    le = joblib.load(out_dir / "models" / "label_encoder.joblib")
    model_path = out_dir / "models" / "mlp_best.keras"
    if not model_path.exists():
        # 백업: 최종 모델이라도 사용
        model_path = out_dir / "models" / "mlp_final.keras"
    model = load_model(model_path)

    # 분할
    X_te, y_te = split_Xy(te)
    X_va, y_va = split_Xy(va)

    # 스케일링 & 인코딩
    X_tes = scaler.transform(X_te)
    X_vas = scaler.transform(X_va)
    y_te_enc = le.transform(y_te)
    y_va_enc = le.transform(y_va)
    labels = list(le.classes_)

    # 예측 (test 기준 리포트 생성)
    proba_te = model.predict(X_tes, verbose=0)
    y_pred_te = proba_te.argmax(axis=1)

    # === Overall metrics (test & val) 저장 ===
    overall = {
        "test_accuracy": float(accuracy_score(y_te_enc, y_pred_te)),
        "test_macroF1": float(f1_score(y_te_enc, y_pred_te, average="macro")),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_te_enc, y_pred_te)),
    }
    # 참고용으로 val도 같이 기록
    proba_va = model.predict(X_vas, verbose=0)
    y_pred_va = proba_va.argmax(axis=1)
    overall.update({
        "val_accuracy": float(accuracy_score(y_va_enc, y_pred_va)),
        "val_macroF1": float(f1_score(y_va_enc, y_pred_va, average="macro")),
        "val_balanced_accuracy": float(balanced_accuracy_score(y_va_enc, y_pred_va)),
    })
    with open(rep_dir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    # === Confusion matrices (test) ===
    cm = confusion_matrix(y_te_enc, y_pred_te, labels=list(range(len(labels))))
    np.savetxt(rep_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    _, cmn = save_cm(cm, labels, rep_dir / "cm_raw.png", rep_dir / "cm_norm.png")
    np.savetxt(rep_dir / "confusion_matrix_norm.csv", cmn, fmt="%.6f", delimiter=",")

    # === Per-class (test): 기본 지표 + OVA ===
    rep = classification_report(y_te_enc, y_pred_te, target_names=labels, output_dict=True, zero_division=0)

    per_class_rows = []
    tot = int(cm.sum())
    for i, name in enumerate(labels):
        d = rep.get(name, {})
        TP = int(cm[i, i])
        FN = int(cm[i, :].sum() - TP)
        FP = int(cm[:, i].sum() - TP)
        TN = int(tot - TP - FN - FP)

        acc_ova = (TP + TN) / max(tot, 1)
        TPR     = TP / max(TP + FN, 1)   # recall
        TNR     = TN / max(TN + FP, 1)
        bal_acc = 0.5 * (TPR + TNR)

        per_class_rows.append({
            "label": name,
            "precision": float(d.get("precision", 0.0)),
            "recall": float(d.get("recall", 0.0)),
            "f1": float(d.get("f1-score", 0.0)),
            "support": int(d.get("support", 0)),
            "accuracy_ova": float(acc_ova),
            "balanced_accuracy_ova": float(bal_acc),
            "tp": TP, "fp": FP, "fn": FN, "tn": TN
        })

    # CSV 저장
    pd.DataFrame(per_class_rows).to_csv(rep_dir / "per_class_report.csv", index=False)

    # TXT 저장 (기본 리포트 + OVA + BENIGN 요약)
    with open(rep_dir / "classification_report.txt", "w", encoding="utf-8", errors="replace") as f:
        f.write(classification_report(y_te_enc, y_pred_te, target_names=labels, zero_division=0))

        f.write("\n\n[Per-class OVA metrics]\n")
        f.write("label,accuracy_ova,balanced_accuracy_ova,support\n")
        for r in per_class_rows:
            f.write(f"{r['label']},{r['accuracy_ova']:.6f},{r['balanced_accuracy_ova']:.6f},{r['support']}\n")

        if "BENIGN" in labels:
            bidx = labels.index("BENIGN")
            TP_b = int(cm[bidx, bidx])
            FN_b = int(cm[bidx, :].sum() - TP_b)   # BENIGN → 공격 (정상을 공격으로)
            FP_b = int(cm[:, bidx].sum() - TP_b)   # 공격 → BENIGN (공격을 정상으로)
            f.write("\n[BENIGN errors]\n")
            f.write(f"False Positives (attack→benign): {FP_b}\n")
            f.write(f"False Negatives (benign→attack): {FN_b}\n")

    print("[evaluate] reports written to", rep_dir)


if __name__ == "__main__":
    main()