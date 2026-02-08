from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import joblib
import numpy as np
import pandas as pd

from src.models.rf import evaluate_rf, train_rf
from src.utils.config import DATA_PROCESSED_DIR, METRICS_DIR, MODELS_DIR, RF_PARAMS


FUSION_FEATURES = ["crescent_prob", "fibrosis_prob", "ePWV", "SII", "24h-UP", "eGFR"]
CLINICAL_FEATURES = ["ePWV", "SII", "24h-UP", "eGFR"]
FUSION_RF_PARAMS = {
    **RF_PARAMS,
    "n_estimators": 60,
    "max_depth": 1,
    "min_samples_leaf": 20,
    "max_features": "sqrt",
}



def build_dataset() -> pd.DataFrame:
    internal_df = pd.read_csv(DATA_PROCESSED_DIR / "internal_split.csv")
    external_df = pd.read_csv(DATA_PROCESSED_DIR / "external_data.csv")
    preds_df = pd.read_csv(DATA_PROCESSED_DIR / "patient_image_preds.csv")
    oof_path = DATA_PROCESSED_DIR / "patient_image_preds_oof.csv"
    oof_df = pd.read_csv(oof_path) if oof_path.exists() else None

    internal_df = internal_df.rename(columns={"病理号": "patient_id"})
    external_df = external_df.rename(columns={"病理号": "patient_id"})
    internal_df["patient_id"] = internal_df["patient_id"].astype(str)
    external_df["patient_id"] = external_df["patient_id"].astype(str)
    preds_df["patient_id"] = preds_df["patient_id"].astype(str)
    if oof_df is not None:
        oof_df["patient_id"] = oof_df["patient_id"].astype(str)

    preds_df = preds_df.drop(columns=[col for col in ["split"] if col in preds_df.columns])
    if oof_df is not None and "split" in oof_df.columns:
        oof_df = oof_df.drop(columns=["split"])
    internal = pd.merge(internal_df, preds_df, on="patient_id", how="inner")
    if oof_df is not None:
        internal = pd.merge(internal, oof_df, on="patient_id", how="left")
    external = pd.merge(external_df, preds_df, on="patient_id", how="inner")

    internal["split"] = internal["split"].fillna("train")
    external["split"] = "external"

    all_df = pd.concat([internal, external], ignore_index=True)
    return all_df



def _select_prob(df: pd.DataFrame, name: str) -> np.ndarray:
    base_col = f"{name}_prob"
    oof_col = f"{name}_prob_oof"
    if oof_col in df.columns:
        return df[oof_col].fillna(df[base_col]).values
    return df[base_col].values



def main() -> None:
    data = build_dataset()
    data = data.dropna(subset=FUSION_FEATURES + ["1yearegfr"])

    train_df = data[data["split"] == "train"].copy()
    val_df = data[data["split"] == "val"].copy()
    ext_df = data[data["split"] == "external"].copy()

    y_train = train_df["1yearegfr"].values
    y_val = val_df["1yearegfr"].values
    y_ext = ext_df["1yearegfr"].values

    crescent_train = _select_prob(train_df, "crescent")
    fibrosis_train = _select_prob(train_df, "fibrosis")
    crescent_val = _select_prob(val_df, "crescent")
    fibrosis_val = _select_prob(val_df, "fibrosis")
    crescent_ext = _select_prob(ext_df, "crescent")
    fibrosis_ext = _select_prob(ext_df, "fibrosis")

    X_train = np.column_stack([crescent_train, fibrosis_train, train_df[CLINICAL_FEATURES].values])
    X_val = np.column_stack([crescent_val, fibrosis_val, val_df[CLINICAL_FEATURES].values])
    X_ext = np.column_stack([crescent_ext, fibrosis_ext, ext_df[CLINICAL_FEATURES].values])

    model = train_rf(X_train, y_train, FUSION_RF_PARAMS)
    train_metrics = evaluate_rf(model, X_train, y_train)
    val_metrics = evaluate_rf(model, X_val, y_val)
    ext_metrics = evaluate_rf(model, X_ext, y_ext)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_fusion_model.joblib"
    joblib.dump(model, model_path)

    # 临床变量对照模型
    clinical_model = train_rf(train_df[CLINICAL_FEATURES].values, y_train, RF_PARAMS)
    clinical_train = evaluate_rf(clinical_model, train_df[CLINICAL_FEATURES].values, y_train)
    clinical_val = evaluate_rf(clinical_model, val_df[CLINICAL_FEATURES].values, y_val)
    clinical_ext = evaluate_rf(clinical_model, ext_df[CLINICAL_FEATURES].values, y_ext)
    clinical_path = MODELS_DIR / "rf_clinical_model.joblib"
    joblib.dump(clinical_model, clinical_path)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    oof_used = "crescent_prob_oof" in train_df.columns or "fibrosis_prob_oof" in train_df.columns
    summary = {
        "features": [
            "crescent_prob_oof" if "crescent_prob_oof" in train_df.columns else "crescent_prob",
            "fibrosis_prob_oof" if "fibrosis_prob_oof" in train_df.columns else "fibrosis_prob",
            *CLINICAL_FEATURES,
        ],
        "best_params": FUSION_RF_PARAMS,
        "image_variant": {"name": "oof_mean_prob" if oof_used else "mean_prob"},
        "oof_used": bool(oof_used),
        "train": train_metrics,
        "val": val_metrics,
        "external": ext_metrics,
        "model_path": str(model_path.resolve()),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "external_samples": int(len(ext_df)),
        "clinical": {
            "features": CLINICAL_FEATURES,
            "train": clinical_train,
            "val": clinical_val,
            "external": clinical_ext,
            "model_path": str(clinical_path.resolve()),
        },
    }
    with open(METRICS_DIR / "rf_fusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("融合 RF 模型训练完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
