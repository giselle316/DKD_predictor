from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import (
    DATA_PROCESSED_DIR,
    EXCEL_PATH,
    IMAGE_DIRS,
    LOGS_DIR,
    SEED,
    TRAIN_VAL_SPLIT,
    USE_TIF_IMAGES,
)
from src.utils.io import extract_patient_id, is_image_file, load_sheet, prepare_labels
from src.utils.seed import set_seed


def choose_split_seed(df: pd.DataFrame, candidates: range, balance_cols: list[str]) -> int:
    scores = []
    stds = {}
    for col in balance_cols:
        std = df[col].std()
        stds[col] = std if std and std > 0 else 1.0
    for seed in candidates:
        train_ids, val_ids = train_test_split(
            df["病理号"],
            test_size=TRAIN_VAL_SPLIT,
            random_state=seed,
            stratify=df["1yearegfr"],
        )
        train = df[df["病理号"].isin(train_ids)]
        val = df[df["病理号"].isin(val_ids)]
        score = 0.0
        for col in balance_cols:
            score += abs(train[col].mean() - val[col].mean()) / stds[col]
        scores.append((seed, score))
    scores.sort(key=lambda x: x[1])
    return scores[0][0]


def build_image_index(
    image_dir: Path,
    df: pd.DataFrame,
    split_name: str,
    label_col: str,
    split_col: str | None = None,
) -> pd.DataFrame:
    rows = []
    missing_patients = set()
    for path in image_dir.rglob("*"):
        if not path.is_file() or not is_image_file(path):
            continue
        pid = extract_patient_id(path.name)
        if pid not in df["病理号"].values:
            missing_patients.add(pid)
            continue
        row = df.loc[df["病理号"] == pid].iloc[0]
        label = int(row[label_col])
        if split_col is not None:
            split_value = row[split_col]
        else:
            split_value = split_name
        rows.append(
            {
                "image_path": str(path.resolve()),
                "patient_id": pid,
                "label": label,
                "split": split_value,
            }
        )
    if missing_patients:
        print(f"[警告] 目录 {image_dir} 中有 {len(missing_patients)} 个病人不在表格中，已忽略。")
    return pd.DataFrame(rows)


def compute_image_stats(df_images: pd.DataFrame) -> pd.DataFrame:
    stats = df_images.groupby("patient_id").size().reset_index(name="image_count")
    return stats


def main() -> None:
    set_seed(SEED)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    missing_dirs = []
    for key, img_dir in IMAGE_DIRS.items():
        if not img_dir.exists():
            missing_dirs.append(key)
            continue
        if not any(img_dir.rglob("*")):
            missing_dirs.append(key)
    if missing_dirs and USE_TIF_IMAGES:
        raise FileNotFoundError(
            f"未找到已转换的 TIF 图像目录：{missing_dirs}，请先运行 00_convert_images_to_tif.py"
        )

    df_internal = load_sheet(EXCEL_PATH, "建模及内部验证")
    df_external = load_sheet(EXCEL_PATH, "外部验证集")

    df_internal = prepare_labels(df_internal)
    df_external = prepare_labels(df_external)

    # 训练/内部验证拆分（优先保证关键特征分布平衡）
    balance_cols = ["Crescent-shaped_changes", "Interstitial_fibrosis", "ePWV", "SII", "24h-UP", "eGFR"]
    chosen_seed = choose_split_seed(df_internal, range(1, 51), balance_cols)
    train_ids, val_ids = train_test_split(
        df_internal["病理号"],
        test_size=TRAIN_VAL_SPLIT,
        random_state=chosen_seed,
        stratify=df_internal["1yearegfr"],
    )
    df_internal = df_internal.copy()
    df_internal["split"] = df_internal["病理号"].apply(
        lambda x: "train" if x in set(train_ids) else "val"
    )

    # 保存内部拆分
    df_internal.to_csv(DATA_PROCESSED_DIR / "internal_split.csv", index=False)
    df_external.to_csv(DATA_PROCESSED_DIR / "external_data.csv", index=False)

    # 图像索引
    crescent_internal = build_image_index(
        IMAGE_DIRS["crescent_internal"],
        df_internal,
        "internal",
        "Crescent-shaped_changes",
        split_col="split",
    )
    fibrosis_internal = build_image_index(
        IMAGE_DIRS["fibrosis_internal"],
        df_internal,
        "internal",
        "Interstitial_fibrosis",
        split_col="split",
    )
    missing_fibrosis_ids = set(df_internal["病理号"]) - set(fibrosis_internal["patient_id"])
    ftest_added = 0
    if missing_fibrosis_ids:
        fibrosis_test = build_image_index(
            IMAGE_DIRS["fibrosis_test"],
            df_internal,
            "internal",
            "Interstitial_fibrosis",
            split_col="split",
        )
        fibrosis_test = fibrosis_test[fibrosis_test["patient_id"].isin(missing_fibrosis_ids)]
        if not fibrosis_test.empty:
            ftest_added = int(fibrosis_test["patient_id"].nunique())
            fibrosis_internal = pd.concat([fibrosis_internal, fibrosis_test], ignore_index=True)
    crescent_external = build_image_index(
        IMAGE_DIRS["crescent_external"], df_external, "external", "Crescent-shaped_changes"
    )
    fibrosis_external = build_image_index(
        IMAGE_DIRS["fibrosis_external"], df_external, "external", "Interstitial_fibrosis"
    )

    crescent_all = pd.concat([crescent_internal, crescent_external], ignore_index=True)
    fibrosis_all = pd.concat([fibrosis_internal, fibrosis_external], ignore_index=True)

    crescent_all.to_csv(DATA_PROCESSED_DIR / "image_index_crescent.csv", index=False)
    fibrosis_all.to_csv(DATA_PROCESSED_DIR / "image_index_fibrosis.csv", index=False)

    # 统计
    summary = {
        "internal_patients": int(df_internal.shape[0]),
        "external_patients": int(df_external.shape[0]),
        "internal_positive_ratio": float(df_internal["1yearegfr"].mean()),
        "external_positive_ratio": float(df_external["1yearegfr"].mean()),
        "split_seed": int(chosen_seed),
        "train_patients": int((df_internal["split"] == "train").sum()),
        "val_patients": int((df_internal["split"] == "val").sum()),
        "train_positive_ratio": float(df_internal[df_internal["split"] == "train"]["1yearegfr"].mean()),
        "val_positive_ratio": float(df_internal[df_internal["split"] == "val"]["1yearegfr"].mean()),
    }

    # 每个病人图像数量统计
    crescent_counts_internal = compute_image_stats(crescent_internal)
    fibrosis_counts_internal = compute_image_stats(fibrosis_internal)
    crescent_counts_external = compute_image_stats(crescent_external)
    fibrosis_counts_external = compute_image_stats(fibrosis_external)

    crescent_counts_internal.to_csv(DATA_PROCESSED_DIR / "crescent_internal_counts.csv", index=False)
    fibrosis_counts_internal.to_csv(DATA_PROCESSED_DIR / "fibrosis_internal_counts.csv", index=False)
    crescent_counts_external.to_csv(DATA_PROCESSED_DIR / "crescent_external_counts.csv", index=False)
    fibrosis_counts_external.to_csv(DATA_PROCESSED_DIR / "fibrosis_external_counts.csv", index=False)

    summary.update(
        {
            "crescent_internal_image_mean": float(crescent_counts_internal["image_count"].mean())
            if not crescent_counts_internal.empty
            else 0,
            "fibrosis_internal_image_mean": float(fibrosis_counts_internal["image_count"].mean())
            if not fibrosis_counts_internal.empty
            else 0,
            "crescent_external_image_mean": float(crescent_counts_external["image_count"].mean())
            if not crescent_counts_external.empty
            else 0,
            "fibrosis_external_image_mean": float(fibrosis_counts_external["image_count"].mean())
            if not fibrosis_counts_external.empty
            else 0,
            "crescent_internal_missing_patients": int(
                len(set(df_internal["病理号"]) - set(crescent_counts_internal["patient_id"]))
            ),
            "fibrosis_internal_missing_patients": int(
                len(set(df_internal["病理号"]) - set(fibrosis_counts_internal["patient_id"]))
            ),
            "fibrosis_internal_added_from_ftest": ftest_added,
            "crescent_external_missing_patients": int(
                len(set(df_external["病理号"]) - set(crescent_counts_external["patient_id"]))
            ),
            "fibrosis_external_missing_patients": int(
                len(set(df_external["病理号"]) - set(fibrosis_counts_external["patient_id"]))
            ),
        }
    )

    # 缺失统计
    required_cols = ["ePWV", "SII", "24h-UP", "eGFR", "1yearegfr"]
    summary["internal_missing"] = (
        df_internal[required_cols].isna().sum().to_dict()
    )
    summary["external_missing"] = (
        df_external[required_cols].isna().sum().to_dict()
    )

    with open(LOGS_DIR / "data_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("数据检查完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
