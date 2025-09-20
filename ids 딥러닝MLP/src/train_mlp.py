# src/train_mlp.py
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from .config import MAIN_LABEL_COL, SUFFIX_COL, ID_COL, RAW_LABEL_COL
from .callbacks import EpochHistory


# -------------------------
# Model
# -------------------------
def build_mlp(input_dim: int, n_classes: int):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ])
    # 안정화: LR 낮춤
    opt = optimizers.Adam(learning_rate=3e-4)  # 1e-3 -> 3e-4
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# -------------------------
# Data helpers
# -------------------------
def load_parts(data_dir: Path):
    tr = pd.read_parquet(data_dir / "train.parquet")
    va = pd.read_parquet(data_dir / "val.parquet")
    te = pd.read_parquet(data_dir / "test.parquet")
    return tr, va, te

def split_Xy(df: pd.DataFrame):
    X = df.drop(columns=[MAIN_LABEL_COL, SUFFIX_COL, ID_COL, RAW_LABEL_COL], errors="ignore")
    y = df[MAIN_LABEL_COL].astype(str)
    return X, y


# -------------------------
# SMOTE (2x cap with absolute ceiling)
# -------------------------
def maybe_smote_force2x(X, y, max_multiplier=2.0, absolute_cap=15000):
    """
    모든 클래스 2배 증강(상한 포함).
    - target = min(floor(orig * max_multiplier), absolute_cap)
    - orig보다 클 때만 증강
    """
    vc = y.value_counts()
    sampling_strategy = {}
    for cls, cnt in vc.items():
        orig = int(cnt)
        target = min(int(np.floor(orig * max_multiplier)), absolute_cap)
        if target > orig:
            sampling_strategy[str(cls)] = target

    if not sampling_strategy:
        return X, y

    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=42, n_jobs=1)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)  # 512 -> 256
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    # Load
    tr, va, te = load_parts(data_dir)
    X_tr, y_tr = split_Xy(tr)
    X_va, y_va = split_Xy(va)

    # Scale
    scaler = RobustScaler()
    X_trs = scaler.fit_transform(X_tr)
    X_vas = scaler.transform(X_va)

    # Label encode
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_va_enc = le.transform(y_va)

    # ------- SMOTE (전처리 이후, 인코딩 전 y 사용) -------
    
    X_aug, y_aug = maybe_smote_force2x(pd.DataFrame(X_trs, columns=X_tr.columns), y_tr)
    if X_aug is not None and len(X_aug) > len(X_trs):
        # y_aug는 문자열 라벨 → 인코딩 다시
        y_tr_enc = le.transform(y_aug)
        X_trs = np.asarray(X_aug)

    # ------- class_weight: 반드시 SMOTE 이후에 '한 번만' 계산 + cap 적용 -------
    classes = np.unique(y_tr_enc)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr_enc)
    cw = np.minimum(cw, 30.0)  # cap=30 확실히 적용
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    # Build & train
    model = build_mlp(input_dim=X_trs.shape[1], n_classes=len(le.classes_))

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,              
            min_delta=1e-4,          
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, cooldown=1),
        callbacks.ModelCheckpoint(filepath=str(out_dir / "models" / "mlp_best.keras"),
                                  monitor="val_loss", save_best_only=True),
        EpochHistory(out_dir=out_dir / "reports"),
    ]

    hist = model.fit(
        X_trs, y_tr_enc,
        validation_data=(X_vas, y_va_enc),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=2,
        callbacks=cb
    )

    # Save artifacts
    import joblib
    joblib.dump(scaler, out_dir / "models" / "scaler.joblib")
    joblib.dump(le, out_dir / "models" / "label_encoder.joblib")
    model.save(out_dir / "models" / "mlp_final.keras")

    # Save training summary
    with open(out_dir / "reports" / "train_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "epochs_run": len(hist.history["loss"]),
            "final_train_loss": float(hist.history["loss"][-1]),
            "final_val_loss": float(hist.history["val_loss"][-1]),
            "final_train_acc": float(hist.history["accuracy"][-1]),
            "final_val_acc": float(hist.history["val_accuracy"][-1]),
            "class_weight": class_weight
        }, f, indent=2, ensure_ascii=False)

    print("[train_mlp] done.")


if __name__ == "__main__":
    main()