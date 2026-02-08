from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import pandas as pd

from src.eval.image_eval import aggregate_patient, predict_images
from src.utils.config import DATA_PROCESSED_DIR, IMG_SIZE, METRICS_DIR, MODELS_DIR


def _merge_splits(df_patients: pd.DataFrame, internal: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    internal_map = internal.set_index("病理号")["split"].to_dict()
    external_ids = set(external["病理号"].tolist())

    splits = []
    for pid in df_patients["patient_id"]:
        if pid in internal_map:
            splits.append(internal_map[pid])
        elif pid in external_ids:
            splits.append("external")
        else:
            splits.append("unknown")
    df_patients["split"] = splits
    return df_patients


def main() -> None:
    crescent_csv = DATA_PROCESSED_DIR / "image_index_crescent.csv"
    fibrosis_csv = DATA_PROCESSED_DIR / "image_index_fibrosis.csv"

    internal_df = pd.read_csv(DATA_PROCESSED_DIR / "internal_split.csv")
    external_df = pd.read_csv(DATA_PROCESSED_DIR / "external_data.csv")

    crescent_model = MODELS_DIR / "crescent_resnet18_best.pt"
    fibrosis_model = MODELS_DIR / "fibrosis_resnet18_best.pt"

    if not crescent_model.exists() or not fibrosis_model.exists():
        raise FileNotFoundError("未找到图像模型权重，请先训练 Crescent/Fibrosis 模型")

    print("开始对 Crescent 图像进行预测...")
    crescent_pred = predict_images(crescent_csv, crescent_model, IMG_SIZE)
    crescent_pred.to_csv(DATA_PROCESSED_DIR / "preds_crescent_images.csv", index=False)

    print("开始对 Fibrosis 图像进行预测...")
    fibrosis_pred = predict_images(fibrosis_csv, fibrosis_model, IMG_SIZE)
    fibrosis_pred.to_csv(DATA_PROCESSED_DIR / "preds_fibrosis_images.csv", index=False)

    crescent_patient = aggregate_patient(crescent_pred, min_positive=2, threshold=0.5)
    fibrosis_patient = aggregate_patient(fibrosis_pred, min_positive=3, threshold=0.5)

    crescent_patient = _merge_splits(crescent_patient, internal_df, external_df)
    fibrosis_patient = _merge_splits(fibrosis_patient, internal_df, external_df)

    crescent_patient = crescent_patient.rename(
        columns={
            "mean_prob": "crescent_prob",
            "logit_mean_prob": "crescent_logit_prob",
            "median_prob": "crescent_median_prob",
            "std_prob": "crescent_std_prob",
            "min_prob": "crescent_min_prob",
            "max_prob": "crescent_max_prob",
            "pos_ratio": "crescent_pos_ratio",
            "entropy_mean": "crescent_entropy",
            "pred": "crescent_pred",
        }
    )
    fibrosis_patient = fibrosis_patient.rename(
        columns={
            "mean_prob": "fibrosis_prob",
            "logit_mean_prob": "fibrosis_logit_prob",
            "median_prob": "fibrosis_median_prob",
            "std_prob": "fibrosis_std_prob",
            "min_prob": "fibrosis_min_prob",
            "max_prob": "fibrosis_max_prob",
            "pos_ratio": "fibrosis_pos_ratio",
            "entropy_mean": "fibrosis_entropy",
            "pred": "fibrosis_pred",
        }
    )

    merged = pd.merge(crescent_patient, fibrosis_patient, on=["patient_id", "split"], how="outer")
    merged.to_csv(DATA_PROCESSED_DIR / "patient_image_preds.csv", index=False)

    summary = {
        "crescent_patients": int(crescent_patient.shape[0]),
        "fibrosis_patients": int(fibrosis_patient.shape[0]),
        "merged_patients": int(merged.shape[0]),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / "image_prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("图像预测完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
