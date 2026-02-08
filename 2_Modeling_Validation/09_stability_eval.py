from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.models.rf import train_rf
from src.utils.config import DATA_PROCESSED_DIR, METRICS_DIR, FIGURES_DIR, RF_PARAMS
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
    internal = pd.merge(internal_df, preds_df, on="patient_id", how="inner")
    if oof_df is not None:
        if "split" in oof_df.columns:
            oof_df = oof_df.drop(columns=["split"])
        internal = pd.merge(internal, oof_df, on="patient_id", how="left")
    external = pd.merge(external_df, preds_df, on="patient_id", how="inner")

    internal["split"] = internal["split"].fillna("train")
    external["split"] = "external"

    internal["crescent_prob_fusion"] = internal["crescent_prob_oof"].fillna(
        internal["crescent_prob"]
    ) if "crescent_prob_oof" in internal.columns else internal["crescent_prob"]
    internal["fibrosis_prob_fusion"] = internal["fibrosis_prob_oof"].fillna(
        internal["fibrosis_prob"]
    ) if "fibrosis_prob_oof" in internal.columns else internal["fibrosis_prob"]
    external["crescent_prob_fusion"] = external["crescent_prob"]
    external["fibrosis_prob_fusion"] = external["fibrosis_prob"]

    all_df = pd.concat([internal, external], ignore_index=True)
    return all_df


def _bootstrap_auc(y_true: np.ndarray, y_prob: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return {"auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    aucs = np.asarray(aucs, dtype=float)
    return {
        "auc": float(np.mean(aucs)),
        "ci_low": float(np.percentile(aucs, 2.5)),
        "ci_high": float(np.percentile(aucs, 97.5)),
        "n": int(len(aucs)),
    }


def main() -> None:
    setup_matplotlib()
    data = build_dataset()

    with open(METRICS_DIR / "rf_stage_summary.json", "r", encoding="utf-8") as f:
        stage_summary = json.load(f)
    with open(METRICS_DIR / "rf_fusion_summary.json", "r", encoding="utf-8") as f:
        fusion_summary = json.load(f)

    stage_features = stage_summary.get("features", CLINICAL_FEATURES)
    stage_params = stage_summary.get("best_params", RF_PARAMS)
    fusion_features = ["crescent_prob_fusion", "fibrosis_prob_fusion", *CLINICAL_FEATURES]
    fusion_params = fusion_summary.get("best_params", RF_PARAMS)
    clinical_features = fusion_summary.get("clinical", {}).get("features", CLINICAL_FEATURES)
    clinical_params = fusion_summary.get("clinical", {}).get("best_params", RF_PARAMS)

    data = data.dropna(
        subset=stage_features + clinical_features + fusion_features + ["1yearegfr"]
    )

    internal = data[data["split"].isin(["train", "val"])].copy()
    X_internal = internal[stage_features].values
    y_internal = internal["1yearegfr"].values

    # 多随机划分稳定性
    random_seeds = list(range(1, 21))
    random_results = []
    for seed in random_seeds:
        X_train, X_val, y_train, y_val = train_test_split(
            X_internal,
            y_internal,
            test_size=0.2,
            random_state=seed,
            stratify=y_internal,
        )
        model = train_rf(X_train, y_train, stage_params)
        prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob)
        random_results.append({"seed": seed, "auc": float(auc)})

    random_df = pd.DataFrame(random_results)
    random_path = METRICS_DIR / "stage_stability_random_splits.csv"
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    random_df.to_csv(random_path, index=False)

    # K-fold 交叉验证稳定性
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_internal, y_internal), start=1):
        model = train_rf(X_internal[train_idx], y_internal[train_idx], stage_params)
        prob = model.predict_proba(X_internal[val_idx])[:, 1]
        auc = roc_auc_score(y_internal[val_idx], prob)
        cv_results.append({"fold": fold, "auc": float(auc)})

    cv_df = pd.DataFrame(cv_results)
    cv_path = METRICS_DIR / "stage_stability_cv.csv"
    cv_df.to_csv(cv_path, index=False)

    # Bootstrap AUC 置信区间（基于主拆分）
    train_df = data[data["split"] == "train"].copy()
    val_df = data[data["split"] == "val"].copy()
    ext_df = data[data["split"] == "external"].copy()

    model = train_rf(train_df[stage_features].values, train_df["1yearegfr"].values, stage_params)
    val_prob = model.predict_proba(val_df[stage_features].values)[:, 1]
    ext_prob = model.predict_proba(ext_df[stage_features].values)[:, 1]

    val_ci = _bootstrap_auc(val_df["1yearegfr"].values, val_prob, n_boot=2000, seed=42)
    ext_ci = _bootstrap_auc(ext_df["1yearegfr"].values, ext_prob, n_boot=2000, seed=43)

    stage_summary = {
        "random_split": {
            "n": len(random_df),
            "mean_auc": float(random_df["auc"].mean()),
            "std_auc": float(random_df["auc"].std(ddof=1)),
        },
        "cv": {
            "n": len(cv_df),
            "mean_auc": float(cv_df["auc"].mean()),
            "std_auc": float(cv_df["auc"].std(ddof=1)),
        },
        "bootstrap": {
            "val": val_ci,
            "external": ext_ci,
        },
    }
    stage_summary_path = METRICS_DIR / "stage_stability_summary.json"
    with open(stage_summary_path, "w", encoding="utf-8") as f:
        json.dump(stage_summary, f, ensure_ascii=False, indent=2)

    # 可视化：随机划分与 CV AUC 分布
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.8, 4.2))
    plt.boxplot([random_df["auc"].values, cv_df["auc"].values], labels=["随机划分", "K-fold CV"])
    plt.ylabel("AUC")
    plt.title("分阶段模型稳定性评估")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage_auc_stability.png", dpi=300)
    plt.close()

    # 可视化：Bootstrap CI
    plt.figure(figsize=(6.8, 4.2))
    labels = ["内部验证", "外部验证"]
    means = [val_ci["auc"], ext_ci["auc"]]
    errors = [
        [val_ci["auc"] - val_ci["ci_low"], ext_ci["auc"] - ext_ci["ci_low"]],
        [val_ci["ci_high"] - val_ci["auc"], ext_ci["ci_high"] - ext_ci["auc"]],
    ]
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errors, capsize=6, color=["#DD8452", "#55A868"])
    plt.xticks(x, labels)
    plt.ylim(0.5, 1.0)
    plt.ylabel("AUC")
    plt.title("分阶段模型 AUC Bootstrap 95% CI")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage_auc_bootstrap.png", dpi=300)
    plt.close()

    # Clinical stability
    X_internal_clin = internal[clinical_features].values
    random_results = []
    for seed in random_seeds:
        X_train, X_val, y_train, y_val = train_test_split(
            X_internal_clin,
            y_internal,
            test_size=0.2,
            random_state=seed,
            stratify=y_internal,
        )
        model = train_rf(X_train, y_train, clinical_params)
        prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob)
        random_results.append({"seed": seed, "auc": float(auc)})
    clinical_random_df = pd.DataFrame(random_results)
    clinical_random_path = METRICS_DIR / "clinical_stability_random_splits.csv"
    clinical_random_df.to_csv(clinical_random_path, index=False)

    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_internal_clin, y_internal), start=1):
        model = train_rf(X_internal_clin[train_idx], y_internal[train_idx], clinical_params)
        prob = model.predict_proba(X_internal_clin[val_idx])[:, 1]
        auc = roc_auc_score(y_internal[val_idx], prob)
        cv_results.append({"fold": fold, "auc": float(auc)})
    clinical_cv_df = pd.DataFrame(cv_results)
    clinical_cv_path = METRICS_DIR / "clinical_stability_cv.csv"
    clinical_cv_df.to_csv(clinical_cv_path, index=False)

    model = train_rf(train_df[clinical_features].values, train_df["1yearegfr"].values, clinical_params)
    val_prob = model.predict_proba(val_df[clinical_features].values)[:, 1]
    ext_prob = model.predict_proba(ext_df[clinical_features].values)[:, 1]
    val_ci = _bootstrap_auc(val_df["1yearegfr"].values, val_prob, n_boot=2000, seed=52)
    ext_ci = _bootstrap_auc(ext_df["1yearegfr"].values, ext_prob, n_boot=2000, seed=53)

    clinical_summary = {
        "random_split": {
            "n": len(clinical_random_df),
            "mean_auc": float(clinical_random_df["auc"].mean()),
            "std_auc": float(clinical_random_df["auc"].std(ddof=1)),
        },
        "cv": {
            "n": len(clinical_cv_df),
            "mean_auc": float(clinical_cv_df["auc"].mean()),
            "std_auc": float(clinical_cv_df["auc"].std(ddof=1)),
        },
        "bootstrap": {
            "val": val_ci,
            "external": ext_ci,
        },
    }
    clinical_summary_path = METRICS_DIR / "clinical_stability_summary.json"
    with open(clinical_summary_path, "w", encoding="utf-8") as f:
        json.dump(clinical_summary, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(6.8, 4.2))
    plt.boxplot([clinical_random_df["auc"].values, clinical_cv_df["auc"].values], labels=["随机划分", "K-fold CV"])
    plt.ylabel("AUC")
    plt.title("临床模型稳定性评估")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clinical_auc_stability.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6.8, 4.2))
    labels = ["内部验证", "外部验证"]
    means = [val_ci["auc"], ext_ci["auc"]]
    errors = [
        [val_ci["auc"] - val_ci["ci_low"], ext_ci["auc"] - ext_ci["ci_low"]],
        [val_ci["ci_high"] - val_ci["auc"], ext_ci["ci_high"] - ext_ci["auc"]],
    ]
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errors, capsize=6, color=["#4C72B0", "#55A868"])
    plt.xticks(x, labels)
    plt.ylim(0.5, 1.0)
    plt.ylabel("AUC")
    plt.title("临床模型 AUC Bootstrap 95% CI")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "clinical_auc_bootstrap.png", dpi=300)
    plt.close()

    # Fusion stability
    X_internal_fusion = internal[fusion_features].values
    random_results = []
    for seed in random_seeds:
        X_train, X_val, y_train, y_val = train_test_split(
            X_internal_fusion,
            y_internal,
            test_size=0.2,
            random_state=seed,
            stratify=y_internal,
        )
        model = train_rf(X_train, y_train, fusion_params)
        prob = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, prob)
        random_results.append({"seed": seed, "auc": float(auc)})
    fusion_random_df = pd.DataFrame(random_results)
    fusion_random_path = METRICS_DIR / "fusion_stability_random_splits.csv"
    fusion_random_df.to_csv(fusion_random_path, index=False)

    cv_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_internal_fusion, y_internal), start=1):
        model = train_rf(X_internal_fusion[train_idx], y_internal[train_idx], fusion_params)
        prob = model.predict_proba(X_internal_fusion[val_idx])[:, 1]
        auc = roc_auc_score(y_internal[val_idx], prob)
        cv_results.append({"fold": fold, "auc": float(auc)})
    fusion_cv_df = pd.DataFrame(cv_results)
    fusion_cv_path = METRICS_DIR / "fusion_stability_cv.csv"
    fusion_cv_df.to_csv(fusion_cv_path, index=False)

    model = train_rf(train_df[fusion_features].values, train_df["1yearegfr"].values, fusion_params)
    val_prob = model.predict_proba(val_df[fusion_features].values)[:, 1]
    ext_prob = model.predict_proba(ext_df[fusion_features].values)[:, 1]
    val_ci = _bootstrap_auc(val_df["1yearegfr"].values, val_prob, n_boot=2000, seed=62)
    ext_ci = _bootstrap_auc(ext_df["1yearegfr"].values, ext_prob, n_boot=2000, seed=63)

    fusion_summary = {
        "random_split": {
            "n": len(fusion_random_df),
            "mean_auc": float(fusion_random_df["auc"].mean()),
            "std_auc": float(fusion_random_df["auc"].std(ddof=1)),
        },
        "cv": {
            "n": len(fusion_cv_df),
            "mean_auc": float(fusion_cv_df["auc"].mean()),
            "std_auc": float(fusion_cv_df["auc"].std(ddof=1)),
        },
        "bootstrap": {
            "val": val_ci,
            "external": ext_ci,
        },
    }
    fusion_summary_path = METRICS_DIR / "fusion_stability_summary.json"
    with open(fusion_summary_path, "w", encoding="utf-8") as f:
        json.dump(fusion_summary, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(6.8, 4.2))
    plt.boxplot([fusion_random_df["auc"].values, fusion_cv_df["auc"].values], labels=["随机划分", "K-fold CV"])
    plt.ylabel("AUC")
    plt.title("融合模型稳定性评估")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fusion_auc_stability.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6.8, 4.2))
    labels = ["内部验证", "外部验证"]
    means = [val_ci["auc"], ext_ci["auc"]]
    errors = [
        [val_ci["auc"] - val_ci["ci_low"], ext_ci["auc"] - ext_ci["ci_low"]],
        [val_ci["ci_high"] - val_ci["auc"], ext_ci["ci_high"] - ext_ci["auc"]],
    ]
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=errors, capsize=6, color=["#55A868", "#C44E52"])
    plt.xticks(x, labels)
    plt.ylim(0.5, 1.0)
    plt.ylabel("AUC")
    plt.title("融合模型 AUC Bootstrap 95% CI")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fusion_auc_bootstrap.png", dpi=300)
    plt.close()

    print("稳定性评估完成：")
    print(f"- 随机划分结果：{random_path}")
    print(f"- 交叉验证结果：{cv_path}")
    print(f"- 统计摘要：{stage_summary_path}")
    print(f"- 临床随机划分：{clinical_random_path}")
    print(f"- 临床交叉验证：{clinical_cv_path}")
    print(f"- 临床摘要：{clinical_summary_path}")
    print(f"- 融合随机划分：{fusion_random_path}")
    print(f"- 融合交叉验证：{fusion_cv_path}")
    print(f"- 融合摘要：{fusion_summary_path}")


if __name__ == "__main__":
    main()
