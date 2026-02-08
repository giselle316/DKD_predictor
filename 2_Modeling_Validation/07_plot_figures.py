from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from src.utils.config import DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR
from src.utils.plot_style import setup_matplotlib


CLINICAL_FEATURES = ["ePWV", "SII", "24h-UP", "eGFR"]


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


def plot_auc_comparison(stage_summary: dict, fusion_summary: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["临床模型", "分阶段模型", "融合模型"]
    val_aucs = [
        fusion_summary["clinical"]["val"]["auc"],
        stage_summary["val"]["auc"],
        fusion_summary["val"]["auc"],
    ]
    ext_aucs = [
        fusion_summary["clinical"]["external"]["auc"],
        stage_summary["external"]["auc"],
        fusion_summary["external"]["auc"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.bar(x - width / 2, val_aucs, width, label="内部验证 AUC", color="#4C72B0")
    ax.bar(x + width / 2, ext_aucs, width, label="外部验证 AUC", color="#55A868")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    ax.grid(axis="y")

    # 轴区域居中，左右留白一致，图例放在右侧留白区
    ax.set_position([0.19, 0.12, 0.62, 0.78])
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.83, 0.86),
    )

    plt.savefig(FIGURES_DIR / "auc_comparison.png", dpi=300)
    plt.close()


def plot_pr_auc_comparison(stage_summary: dict, fusion_summary: dict) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["临床模型", "分阶段模型", "融合模型"]
    val_pr = [
        fusion_summary["clinical"]["val"]["pr_auc"],
        stage_summary["val"]["pr_auc"],
        fusion_summary["val"]["pr_auc"],
    ]
    ext_pr = [
        fusion_summary["clinical"]["external"]["pr_auc"],
        stage_summary["external"]["pr_auc"],
        fusion_summary["external"]["pr_auc"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.bar(x - width / 2, val_pr, width, label="内部验证 PR-AUC", color="#4C72B0")
    ax.bar(x + width / 2, ext_pr, width, label="外部验证 PR-AUC", color="#55A868")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("PR-AUC")
    ax.grid(axis="y")

    ax.set_position([0.19, 0.12, 0.62, 0.78])
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.83, 0.86),
    )

    plt.savefig(FIGURES_DIR / "pr_auc_comparison.png", dpi=300)
    plt.close()


def plot_basic_metrics(metrics_df: pd.DataFrame, split: str, out_name: str, title: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    subset = metrics_df[metrics_df["split"] == split]
    metrics = ["accuracy", "precision", "recall", "f1"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score"]
    model_order = ["临床模型", "分阶段模型", "融合模型"]

    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    for i, model in enumerate(model_order):
        row = subset[subset["model"] == model]
        if row.empty:
            continue
        values = row.iloc[0][metrics].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * width, values, width, label=model)

    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("指标值")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)

    # 轴区域居中，右侧留白放图例，避免整体左移
    ax.set_position([0.19, 0.12, 0.62, 0.78])
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.83, 0.86),
    )

    plt.savefig(FIGURES_DIR / out_name, dpi=300)
    plt.close()


def plot_roc_curves(
    data: pd.DataFrame,
    splits: list[tuple[str, str, str]],
    stage_features: list[str],
    fusion_features: list[str],
    clinical_features: list[str],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = {
        "临床模型": (MODELS_DIR / "rf_clinical_model.joblib", clinical_features, "#4C72B0"),
        "分阶段模型": (MODELS_DIR / "rf_stage_model.joblib", stage_features, "#DD8452"),
        "融合模型": (MODELS_DIR / "rf_fusion_model.joblib", fusion_features, "#55A868"),
    }

    for split_name, title, out_name in splits:
        subset = data[data["split"] == split_name]
        y_true = subset["1yearegfr"].values
        plt.figure(figsize=(6, 5))
        for name, (model_path, features, color) in models.items():
            model = joblib.load(model_path)
            if name == "融合模型":
                cres = _select_prob(subset, "crescent")
                fib = _select_prob(subset, "fibrosis")
                X = np.column_stack([cres, fib, subset[clinical_features].values])
            else:
                X = subset[features].values
            y_prob = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            # Interpolate for a smoother academic-style curve
            grid = np.linspace(0, 1, 400)
            tpr_smooth = np.interp(grid, fpr, tpr)
            plt.plot(grid, tpr_smooth, label=f"{name} (AUC={roc_auc:.3f})", color=color)
        plt.plot([0, 1], [0, 1], "--", color="#999999", linewidth=1)
        plt.xlabel("假阳性率")
        plt.ylabel("真阳性率")
        plt.title(title)
        plt.legend(frameon=False)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / out_name, dpi=300)
        plt.close()


def plot_pr_curves(
    data: pd.DataFrame,
    splits: list[tuple[str, str, str]],
    stage_features: list[str],
    fusion_features: list[str],
    clinical_features: list[str],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = {
        "临床模型": (MODELS_DIR / "rf_clinical_model.joblib", clinical_features, "#4C72B0"),
        "分阶段模型": (MODELS_DIR / "rf_stage_model.joblib", stage_features, "#DD8452"),
        "融合模型": (MODELS_DIR / "rf_fusion_model.joblib", fusion_features, "#55A868"),
    }

    for split_name, title, out_name in splits:
        subset = data[data["split"] == split_name]
        y_true = subset["1yearegfr"].values
        plt.figure(figsize=(6, 5))
        for name, (model_path, features, color) in models.items():
            model = joblib.load(model_path)
            if name == "融合模型":
                cres = _select_prob(subset, "crescent")
                fib = _select_prob(subset, "fibrosis")
                X = np.column_stack([cres, fib, subset[clinical_features].values])
            else:
                X = subset[features].values
            y_prob = model.predict_proba(X)[:, 1]
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            # Ensure increasing recall for interpolation
            order = np.argsort(recall)
            recall_sorted = recall[order]
            precision_sorted = precision[order]
            grid = np.linspace(0, 1, 400)
            prec_smooth = np.interp(grid, recall_sorted, precision_sorted)
            plt.plot(grid, prec_smooth, label=f"{name} (AP={ap:.3f})", color=color)
        plt.xlabel("召回率")
        plt.ylabel("精确率")
        plt.title(title)
        plt.legend(frameon=False)
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / out_name, dpi=300)
        plt.close()


def build_metrics_table(stage_summary: dict, fusion_summary: dict) -> pd.DataFrame:
    rows = []
    model_map = [
        ("临床模型", fusion_summary.get("clinical", {})),
        ("分阶段模型", stage_summary),
        ("融合模型", fusion_summary),
    ]
    for model_name, summary in model_map:
        for split in ["train", "val", "external"]:
            if split not in summary:
                continue
            metrics = summary[split]
            rows.append(
                {
                    "model": model_name,
                    "split": split,
                    "accuracy": metrics.get("acc"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "roc_auc": metrics.get("roc_auc", metrics.get("auc")),
                    "pr_auc": metrics.get("pr_auc"),
                }
            )
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    setup_matplotlib()

    with open(METRICS_DIR / "rf_stage_summary.json", "r", encoding="utf-8") as f:
        stage_summary = json.load(f)
    with open(METRICS_DIR / "rf_fusion_summary.json", "r", encoding="utf-8") as f:
        fusion_summary = json.load(f)

    plot_auc_comparison(stage_summary, fusion_summary)
    plot_pr_auc_comparison(stage_summary, fusion_summary)

    stage_features = stage_summary.get("features", CLINICAL_FEATURES)
    fusion_features = fusion_summary.get("features", CLINICAL_FEATURES)
    clinical_features = fusion_summary.get("clinical", {}).get("features", CLINICAL_FEATURES)

    data = build_dataset()
    data = data.dropna(
        subset=["crescent_prob", "fibrosis_prob", *clinical_features, "1yearegfr"]
    )
    plot_roc_curves(
        data,
        [
            ("train", "训练集 ROC 曲线", "roc_train.png"),
            ("val", "内部验证 ROC 曲线", "roc_internal.png"),
            ("external", "外部验证 ROC 曲线", "roc_external.png"),
        ],
        stage_features,
        fusion_features,
        clinical_features,
    )
    plot_pr_curves(
        data,
        [
            ("train", "训练集 PR 曲线", "pr_train.png"),
            ("val", "内部验证 PR 曲线", "pr_internal.png"),
            ("external", "外部验证 PR 曲线", "pr_external.png"),
        ],
        stage_features,
        fusion_features,
        clinical_features,
    )

    metrics_df = build_metrics_table(stage_summary, fusion_summary)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(METRICS_DIR / "classification_metrics.csv", index=False)
    plot_basic_metrics(metrics_df, "val", "metrics_internal.png", "内部验证指标对比")
    plot_basic_metrics(metrics_df, "external", "metrics_external.png", "外部验证指标对比")

    print("图像绘制完成，已保存到 figures/；指标表格已保存到 artifacts/metrics/")


if __name__ == "__main__":
    main()
