from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import joblib
import numpy as np
import pandas as pd

from src.models.rf import evaluate_rf, train_rf
from src.utils.config import (
    DATA_PROCESSED_DIR,
    METRICS_DIR,
    MODELS_DIR,
    STAGE_RF_PARAMS,
    STAGE_THRESHOLD,
)


CLINICAL_FEATURES = ["ePWV", "SII", "24h-UP", "eGFR"]
FEATURES = ["crescent_max_prob", "fibrosis_max_prob", *CLINICAL_FEATURES]
IMAGE_VARIANT = {"variant": "max_prob"}


def build_dataset() -> pd.DataFrame:
    internal_df = pd.read_csv(DATA_PROCESSED_DIR / "internal_split.csv")
    external_df = pd.read_csv(DATA_PROCESSED_DIR / "external_data.csv")
    preds_df = pd.read_csv(DATA_PROCESSED_DIR / "patient_image_preds.csv")

    internal_df = internal_df.rename(columns={"病理号": "patient_id"})
    external_df = external_df.rename(columns={"病理号": "patient_id"})
    internal_df["patient_id"] = internal_df["patient_id"].astype(str)
    external_df["patient_id"] = external_df["patient_id"].astype(str)
    preds_df["patient_id"] = preds_df["patient_id"].astype(str)
    preds_df = preds_df.drop(columns=[col for col in ["split"] if col in preds_df.columns])
    internal = pd.merge(internal_df, preds_df, on="patient_id", how="inner")
    external = pd.merge(external_df, preds_df, on="patient_id", how="inner")

    internal["split"] = internal["split"].fillna("train")
    external["split"] = "external"

    all_df = pd.concat([internal, external], ignore_index=True)
    return all_df


def main() -> None:
    data = build_dataset()

    # 过滤缺失特征
    data = data.dropna(subset=FEATURES + ["1yearegfr"])

    train_df = data[data["split"] == "train"].copy()
    val_df = data[data["split"] == "val"].copy()
    ext_df = data[data["split"] == "external"].copy()

    X_train = train_df[FEATURES].values
    y_train = train_df["1yearegfr"].values
    X_val = val_df[FEATURES].values
    y_val = val_df["1yearegfr"].values
    X_ext = ext_df[FEATURES].values
    y_ext = ext_df["1yearegfr"].values

    model = train_rf(X_train, y_train, STAGE_RF_PARAMS)
    train_metrics = evaluate_rf(model, X_train, y_train, threshold=STAGE_THRESHOLD)
    val_metrics = evaluate_rf(model, X_val, y_val, threshold=STAGE_THRESHOLD)
    ext_metrics = evaluate_rf(model, X_ext, y_ext, threshold=STAGE_THRESHOLD)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_stage_model.joblib"
    joblib.dump(model, model_path)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "features": FEATURES,
        "oof_used": False,
        "threshold": float(STAGE_THRESHOLD),
        "best_params": STAGE_RF_PARAMS,
        "image_variant": IMAGE_VARIANT,
        "train": train_metrics,
        "val": val_metrics,
        "external": ext_metrics,
        "model_path": str(model_path.resolve()),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "external_samples": int(len(ext_df)),
    }
    with open(METRICS_DIR / "rf_stage_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("分阶段 RF 模型训练完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
