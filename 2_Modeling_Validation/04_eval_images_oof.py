from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from src.datasets.image_dataset import ImageDatasetFrame
from src.models.resnet18 import build_resnet18
from src.train.train_image import train_model
from src.eval.image_eval import aggregate_patient
from src.utils.config import (
    BATCH_SIZE,
    DATA_PROCESSED_DIR,
    IMG_SIZE,
    LEARNING_RATE,
    EPOCHS,
    NUM_WORKERS,
    PATIENCE,
    WEIGHT_DECAY,
    METRICS_DIR,
)
from src.utils.seed import set_seed
from src.utils.device import get_device


def _build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, val_transform


def _predict_df(model: torch.nn.Module, df: pd.DataFrame, transform: transforms.Compose) -> pd.DataFrame:
    device = get_device()
    model.eval()
    probs = []
    for _, row in df.iterrows():
        img = Image.open(row["image_path"]).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img)
            prob = torch.softmax(logits, dim=1)[:, 1].item()
        probs.append(prob)
    out = df.copy()
    out["prob"] = probs
    return out


def _train_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
) -> torch.nn.Module:
    train_loader = DataLoader(
        ImageDatasetFrame(train_df, transform=train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        ImageDatasetFrame(val_df, transform=val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    model = build_resnet18(num_classes=2)
    train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        save_path=None,
    )
    return model


def _oof_predict(
    df_images: pd.DataFrame,
    patient_labels: pd.Series,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    train_transform, val_transform = _build_transforms()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_rows = []
    for fold, (train_idx, oof_idx) in enumerate(skf.split(patient_labels.index, patient_labels.values), start=1):
        train_patients = patient_labels.index[train_idx]
        oof_patients = patient_labels.index[oof_idx]

        train_labels = patient_labels.loc[train_patients]
        sub_train, sub_val = train_test_split(
            train_patients,
            test_size=0.15,
            random_state=seed + fold,
            stratify=train_labels,
        )

        train_df = df_images[df_images["patient_id"].isin(sub_train)].copy()
        val_df = df_images[df_images["patient_id"].isin(sub_val)].copy()
        oof_df = df_images[df_images["patient_id"].isin(oof_patients)].copy()

        model = _train_fold(train_df, val_df, train_transform, val_transform)
        oof_pred = _predict_df(model, oof_df, val_transform)
        oof_rows.append(oof_pred)

    return pd.concat(oof_rows, ignore_index=True)


def _rename_with_suffix(df: pd.DataFrame, prefix: str, suffix: str) -> pd.DataFrame:
    mapping = {
        "mean_prob": f"{prefix}_prob{suffix}",
        "logit_mean_prob": f"{prefix}_logit_prob{suffix}",
        "median_prob": f"{prefix}_median_prob{suffix}",
        "std_prob": f"{prefix}_std_prob{suffix}",
        "min_prob": f"{prefix}_min_prob{suffix}",
        "max_prob": f"{prefix}_max_prob{suffix}",
        "pos_ratio": f"{prefix}_pos_ratio{suffix}",
        "entropy_mean": f"{prefix}_entropy{suffix}",
        "pred": f"{prefix}_pred{suffix}",
        "num_images": f"{prefix}_num_images{suffix}",
        "positive_images": f"{prefix}_positive_images{suffix}",
    }
    return df.rename(columns=mapping)


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"[设备] 使用 {device}")

    internal_df = pd.read_csv(DATA_PROCESSED_DIR / "internal_split.csv")
    internal_df["patient_id"] = internal_df["病理号"].astype(str)

    crescent_csv = DATA_PROCESSED_DIR / "image_index_crescent.csv"
    fibrosis_csv = DATA_PROCESSED_DIR / "image_index_fibrosis.csv"

    crescent_images = pd.read_csv(crescent_csv)
    fibrosis_images = pd.read_csv(fibrosis_csv)
    crescent_images["patient_id"] = crescent_images["patient_id"].astype(str)
    fibrosis_images["patient_id"] = fibrosis_images["patient_id"].astype(str)

    internal_ids = set(internal_df["patient_id"])
    crescent_images = crescent_images[crescent_images["patient_id"].isin(internal_ids)]
    fibrosis_images = fibrosis_images[fibrosis_images["patient_id"].isin(internal_ids)]

    crescent_labels = crescent_images.groupby("patient_id")["label"].first()
    fibrosis_labels = fibrosis_images.groupby("patient_id")["label"].first()

    print("开始 OOF 预测（Crescent）...")
    crescent_oof = _oof_predict(crescent_images, crescent_labels, n_splits=3, seed=42)
    crescent_patient = aggregate_patient(crescent_oof, min_positive=2, threshold=0.5)
    crescent_patient = _rename_with_suffix(crescent_patient, "crescent", "_oof")

    print("开始 OOF 预测（Fibrosis）...")
    fibrosis_oof = _oof_predict(fibrosis_images, fibrosis_labels, n_splits=3, seed=42)
    fibrosis_patient = aggregate_patient(fibrosis_oof, min_positive=3, threshold=0.5)
    fibrosis_patient = _rename_with_suffix(fibrosis_patient, "fibrosis", "_oof")

    merged = pd.merge(crescent_patient, fibrosis_patient, on="patient_id", how="outer")
    merged = pd.merge(
        internal_df[["patient_id", "split"]],
        merged,
        on="patient_id",
        how="left",
    )

    out_path = DATA_PROCESSED_DIR / "patient_image_preds_oof.csv"
    merged.to_csv(out_path, index=False)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "internal_patients": int(len(internal_ids)),
        "crescent_patients": int(crescent_patient.shape[0]),
        "fibrosis_patients": int(fibrosis_patient.shape[0]),
        "merged_patients": int(merged.shape[0]),
        "oof_folds": 3,
    }
    summary_path = METRICS_DIR / "image_oof_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("OOF 图像预测完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"保存路径：{out_path}")


if __name__ == "__main__":
    main()
