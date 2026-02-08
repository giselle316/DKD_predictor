from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from src.datasets.image_dataset import ImageDataset
from src.models.resnet18 import build_resnet18
from src.train.train_image import train_model
from src.utils.config import (
    BATCH_SIZE,
    EPOCHS,
    IMG_SIZE,
    LEARNING_RATE,
    METRICS_DIR,
    MODELS_DIR,
    NUM_WORKERS,
    PATIENCE,
    WEIGHT_DECAY,
    DATA_PROCESSED_DIR,
)
from src.utils.device import get_device
from src.utils.seed import set_seed


def main() -> None:
    set_seed(42)
    device = get_device()
    print(f"[设备] 使用 {device}")

    csv_path = DATA_PROCESSED_DIR / "image_index_fibrosis.csv"
    if not csv_path.exists():
        raise FileNotFoundError("未找到 image_index_fibrosis.csv，请先运行 01_check_data.py")

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

    train_dataset = ImageDataset(csv_path, split="train", transform=train_transform)
    val_dataset = ImageDataset(csv_path, split="val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_resnet18(num_classes=2)

    save_path = MODELS_DIR / "fibrosis_resnet18_best.pt"
    result = train_model(
        model,
        train_loader,
        val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        save_path=save_path,
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = METRICS_DIR / "fibrosis_train_history.csv"
    result.history.to_csv(history_path, index=False)

    summary = {
        "best_auc": result.best_auc,
        "best_epoch": result.best_epoch,
        "model_path": str(save_path.resolve()),
    }
    with open(METRICS_DIR / "fibrosis_train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Fibrosis 模型训练完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
