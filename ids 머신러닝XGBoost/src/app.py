# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import random
import time
import traceback
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, send_from_directory
from flask_socketio import SocketIO

# XGBoost는 환경에 따라 미설치일 수 있으므로 방어
try:
    import xgboost as xgb
except Exception:
    xgb = None  # 데모 모드 대비

# ──────────────────────────────────────────────────────────────────────────────
# 경로/상수
# ──────────────────────────────────────────────────────────────────────────────
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]                   # 프로젝트 루트
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TEMPLATES_DIR = ROOT / "web"             # index.html이 여기에 있다고 가정
STATIC_DIR = ROOT / "static"
LOGS_DIR = ROOT / "logs"; LOGS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "xgb_multiclass.json"
X_TEST_PQ = DATA_DIR / "X_test.parquet"
Y_TEST_PQ = DATA_DIR / "y_test.parquet"
FEAT_TXT = DATA_DIR / "feature_names.txt"
LABEL_JSON = DATA_DIR / "label_encoder.json"

STREAM_INTERVAL_SEC = 2.0               # 이벤트 발행 간격
RECENT_MAX_ROWS = 200                   # 최근 이벤트 버퍼 크기
STREAM_LOG_PATH = LOGS_DIR / "stream_log.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Flask / Socket.IO
# ──────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
app.config["SECRET_KEY"] = "ids-secret"
socketio = SocketIO(
    app,
    async_mode="threading",
    cors_allowed_origins="*",
    ping_timeout=30,
    ping_interval=10,
    max_http_buffer_size=10_000_000,
)

# ──────────────────────────────────────────────────────────────────────────────
# 전역 상태 (스레드 세이프)
# ──────────────────────────────────────────────────────────────────────────────
lock = Lock()

booster: "xgb.Booster | None" = None     # XGBoost 모델
id2lab: Dict[int, str] = {}              # {클래스ID: 라벨명}
lab2id: Dict[str, int] = {}              # {라벨명: 클래스ID}
model_feats: List[str] = []              # 모델이 기대하는 피처 순서
X_test: "pd.DataFrame | None" = None
y_test: "np.ndarray | None" = None

recent_events = deque(maxlen=RECENT_MAX_ROWS)
per_class_total = defaultdict(int)
per_class_correct = defaultdict(int)
total_count = 0
total_top1_correct = 0
total_top3_correct = 0
demo_mode = False

# ──────────────────────────────────────────────────────────────────────────────
# 유틸 (evaluate/train 스크립트 규격과 호환)                                   # :contentReference[oaicite:6]{index=6}
# ──────────────────────────────────────────────────────────────────────────────
_NON_FEATURE_TOKENS = {
    "precision", "recall", "f1", "f1-score", "support", "accuracy", "macro avg", "weighted avg",
}

def _log(msg: str) -> None:
    print(msg, flush=True)

def clean_feature_list(raw_list: List[str]) -> List[str]:
    """feature_names.txt에서 지표/잡토큰 제거 + 중복 제거."""
    keep, seen = [], set()
    for c in raw_list:
        c = str(c).strip()
        if not c or c.lower() in _NON_FEATURE_TOKENS:
            continue
        if c not in seen:
            keep.append(c); seen.add(c)
    return keep

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """object→수치 변환(+결측은 그대로 두고 이후 보정)."""
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def align_to_model_features(df: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    """
    모델이 기억하는 feature_names 순서에 맞춤:
    - 누락 피처는 0.0으로 추가
    - 모델에 없는 피처는 드롭
    - 순서 정확히 정렬
    """
    X = df.copy()
    exist = set(X.columns)
    add = [c for c in feats if c not in exist]
    for c in add:
        X[c] = 0.0
    drop = [c for c in X.columns if c not in feats]
    if drop:
        X.drop(columns=drop, inplace=True, errors="ignore")
    X = X[feats]
    if add:
        _log(f"[WARN] 테스트셋에 없던 피처 {len(add)}개 0.0으로 추가: {add[:10]}{' ...' if len(add) > 10 else ''}")
    if drop:
        _log(f"[WARN] 모델에 없는 테스트 피처 {len(drop)}개 드롭: {drop[:10]}{' ...' if len(drop) > 10 else ''}")
    return X

def load_label_mapping_from_json() -> None:
    """label_encoder.json → id2lab/lab2id 로드(두 포맷 모두 지원)."""
    global id2lab, lab2id
    if not LABEL_JSON.exists():
        return
    enc = json.loads(LABEL_JSON.read_text(encoding="utf-8"))
    mapping = enc.get("mapping", enc)  # {"원라벨": id} or {"mapping": {...}}
    lab2id = {str(k): int(v) for k, v in mapping.items()}
    id2lab = {int(v): str(k) for k, v in mapping.items()}

def predict_proba_compat(X_df: pd.DataFrame, n_classes: int) -> np.ndarray:
    """
    XGBoost 1.x/2.x 호환 예측:
      1) inplace_predict(float32) 시도
      2) 실패 시 DMatrix.predict 폴백
      3) (n_samples, n_classes) 보장
    """
    if booster is None:
        # 데모 모드: 랜덤 디리클레 분포
        k = max(2, n_classes)
        return np.random.dirichlet(alpha=np.ones(k), size=len(X_df))

    X_np = X_df.to_numpy(dtype=np.float32, copy=False)
    proba = None
    try:
        proba = booster.inplace_predict(X_np, validate_features=False)
    except Exception as e1:
        _log(f"[WARN] inplace_predict 실패 → {type(e1).__name__}: {e1}")
    if proba is None:
        try:
            dmat = xgb.DMatrix(X_np, feature_names=list(X_df.columns))
            proba = booster.predict(dmat)
        except Exception as e2:
            raise RuntimeError(f"DMatrix.predict 실패: {type(e2).__name__}: {e2}")
    proba = np.asarray(proba)
    if proba.ndim == 1:
        proba = proba.reshape(X_np.shape[0], -1)
    if proba.shape[1] != n_classes:
        proba = proba.reshape(X_np.shape[0], n_classes)
        _log(f"[WARN] 예측 차원을 n_classes={n_classes}로 보정")
    return proba

def append_stream_log(ev: Dict) -> None:
    header = not STREAM_LOG_PATH.exists()
    with open(STREAM_LOG_PATH, "a", encoding="utf-8") as f:
        if header:
            f.write(",".join(ev.keys()) + "\n")
        f.write(",".join(str(v) for v in ev.values()) + "\n")

def compute_class_stats():
    """클래스별 카운트/정확도 + 전체 Top-1/Top-3 정확도 집계."""
    stats, dist_labels, dist_counts = [], [], []
    with lock:
        for cid in sorted(id2lab.keys()):
            name = id2lab[cid]
            tot = per_class_total[name]
            cor = per_class_correct[name]
            acc = (cor / tot) if tot > 0 else 0.0
            stats.append({"label": name, "count": tot, "correct": cor, "accuracy": round(acc, 4)})
            dist_labels.append(name); dist_counts.append(tot)
        overall = {
            "total_count": total_count,
            "top1_correct": total_top1_correct,
            "top3_correct": total_top3_correct,
            "top1_accuracy": round((total_top1_correct / total_count) if total_count else 0.0, 4),
            "top3_accuracy": round((total_top3_correct / total_count) if total_count else 0.0, 4),
        }
    return stats, overall, {"labels": dist_labels, "counts": dist_counts}

# ──────────────────────────────────────────────────────────────────────────────
# 로딩(모델/데이터)
# ──────────────────────────────────────────────────────────────────────────────
def load_model_and_data() -> pd.DataFrame:
    """
    반환: 모델 피처 순서로 정렬된 X_test DataFrame.
    - 파일 없으면 데모 모드로 난수 생성(라벨 매핑도 임시).
    - 규격은 build/evaluate와 동일하게 맞춤.                                      # :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}
    """
    global booster, X_test, y_test, model_feats, demo_mode, id2lab, lab2id

    # 1) y_test(정답) 로드
    if Y_TEST_PQ.exists():
        ydf = pd.read_parquet(Y_TEST_PQ)
        col = "label" if "label" in ydf.columns else ydf.columns[0]
        y_test = ydf[col].astype(int).values
        n_classes = int(np.max(y_test)) + 1 if len(y_test) else 2
        if not id2lab:
            id2lab = {i: str(i) for i in range(n_classes)}
            lab2id = {v: k for k, v in id2lab.items()}
    else:
        y_test = None
        id2lab = {i: f"C{i}" for i in range(5)}
        lab2id = {v: k for k, v in id2lab.items()}

    # 2) 라벨 매핑 보강
    load_label_mapping_from_json()  # label_encoder.json 존재 시 원라벨명 되살림  # :contentReference[oaicite:9]{index=9}

    # 3) 모델/테스트셋 로드 (실모드)
    if (xgb is not None) and MODEL_PATH.exists() and X_TEST_PQ.exists():
        X_test = ensure_numeric(pd.read_parquet(X_TEST_PQ))
        booster_local = xgb.Booster(); booster_local.load_model(str(MODEL_PATH))  # :contentReference[oaicite:10]{index=10}
        feats = getattr(booster_local, "feature_names", None)
        model_feats = list(feats) if feats else (
            clean_feature_list(FEAT_TXT.read_text(encoding="utf-8").splitlines()) if FEAT_TXT.exists() else list(X_test.columns)
        )
        X_align = align_to_model_features(X_test, model_feats).astype(np.float32)
        demo_mode = False
        booster = booster_local
        _log("[MODEL] XGBoost 모델 및 테스트셋 로드 완료.")
        return X_align

    # 4) 데모 모드 (파일 일부 없을 때)
    if X_TEST_PQ.exists():
        X_test = ensure_numeric(pd.read_parquet(X_TEST_PQ))
        model_feats = list(X_test.columns)
        X_align = X_test.astype(np.float32).copy()
    else:
        rows = len(y_test) if y_test is not None else 500
        cols = 20
        model_feats = [f"F{i}" for i in range(cols)]
        X_align = pd.DataFrame(np.random.randn(rows, cols), columns=model_feats).astype(np.float32)
    booster = None
    demo_mode = True
    _log("[MODEL] 데모 모드로 동작합니다. (모델/데이터 일부 없음)")
    return X_align

# ──────────────────────────────────────────────────────────────────────────────
# 스트림 루프
# ──────────────────────────────────────────────────────────────────────────────
def stream_loop(X_align: pd.DataFrame) -> None:
    """테스트셋에서 임의 샘플을 뽑아 예측→지표 갱신→소켓 송신."""
    global total_count, total_top1_correct, total_top3_correct
    n = int(getattr(X_align, "shape", (0, 0))[0])
    _log(f"[STREAM] starting... samples={n}")
    if n == 0:
        _log("[STREAM][FATAL] X_align is empty. data/processed 확인 필요.")
        return

    while True:
        try:
            idx = random.randint(0, n - 1)
            x_row = X_align.iloc[[idx]]

            # 정답 ID
            if y_test is not None and idx < len(y_test):
                true_id = int(y_test[idx])
            else:
                true_id = random.randint(0, len(id2lab) - 1)
            true_label = id2lab.get(true_id, str(true_id))

            # 예측
            proba = predict_proba_compat(x_row, n_classes=len(id2lab))
            pred_id = int(np.argmax(proba[0]))
            pred_label = id2lab.get(pred_id, str(pred_id))
            top1_score = float(np.max(proba[0]))

            # Top-3
            top3_idx = np.argsort(proba[0])[-3:][::-1]
            top3_labels = [id2lab.get(int(i), str(int(i))) for i in top3_idx]
            top3_scores = [float(proba[0][i]) for i in top3_idx]

            # 이벤트 구성
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ev = {
                "timestamp": ts,
                "true_label": true_label,
                "pred_label": pred_label,
                "proba_top1": round(top1_score, 6),
                "top1_label": top3_labels[0], "top1_proba": round(top3_scores[0], 6),
                "top2_label": top3_labels[1], "top2_proba": round(top3_scores[1], 6),
                "top3_label": top3_labels[2], "top3_proba": round(top3_scores[2], 6),
            }

            # 전역 집계 갱신
            with lock:
                per_class_total[true_label] += 1
                if pred_id == true_id:
                    per_class_correct[true_label] += 1
                    total_top1_correct += 1
                if int(true_id) in [int(i) for i in top3_idx]:
                    total_top3_correct += 1
                globals()["total_count"] += 1
                recent_events.appendleft(ev)

            append_stream_log(ev)

            # 전송
            socketio.emit("stream_event", ev)
            cls_stats, overall, dist = compute_class_stats()
            socketio.emit("stats", {"per_class": cls_stats, "overall": overall, "distribution": dist})

            time.sleep(STREAM_INTERVAL_SEC)

            # 루프 탈출 조건은 없음(서버가 켜져 있는 동안 계속)
        except Exception as e:
            _log(f"[STREAM][ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()
            time.sleep(1.0)

# ──────────────────────────────────────────────────────────────────────────────
# 라우팅/소켓 이벤트
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # web/index.html을 템플릿으로 사용 (없으면 간단 안내 반환)
    if TEMPLATES_DIR.exists() and (TEMPLATES_DIR / "index.html").exists():
        return render_template("index.html")
    return "<h3>대시보드 템플릿(web/index.html)이 없습니다. /health로 상태를 확인하세요.</h3>"

@app.get("/health")
def health():
    return jsonify({
        "X_test_rows": int(getattr(X_test, "shape", (0, 0))[0]) if X_test is not None else 0,
        "y_test_rows": int(len(y_test) if y_test is not None else 0),
        "model_loaded": bool(booster is not None),
        "labels": len(id2lab),
        "static_exists": (Path(app.static_folder) / "style.css").exists() if app.static_folder else False,
        "demo_mode": demo_mode,
    })

@app.post("/reset")
def reset_stats():
    global total_count, total_top1_correct, total_top3_correct
    with lock:
        total_count = 0
        total_top1_correct = 0
        total_top3_correct = 0
        per_class_total.clear()
        per_class_correct.clear()
        recent_events.clear()
    cls_stats, overall, dist = compute_class_stats()
    socketio.emit("stats", {"per_class": cls_stats, "overall": overall, "distribution": dist})
    return {"ok": True}

@app.route("/logs/<path:filename>")
def download_log(filename):
    return send_from_directory(LOGS_DIR, filename, as_attachment=True)

@socketio.on("connect")
def on_connect():
    cls_stats, overall, dist = compute_class_stats()
    socketio.emit("stats", {"per_class": cls_stats, "overall": overall, "distribution": dist})

@socketio.on("snapshot")
def on_snapshot():
    cls_stats, overall, dist = compute_class_stats()
    socketio.emit("stats", {"per_class": cls_stats, "overall": overall, "distribution": dist})

# ──────────────────────────────────────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────────────────────────────────────
def main():
    X_align = load_model_and_data()
    _log("[INFO] 대시보드 시작 → http://127.0.0.1:5000")
    socketio.start_background_task(stream_loop, X_align)
    socketio.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()