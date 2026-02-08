from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.config import DATA_PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR
from src.utils.plot_style import setup_matplotlib


FEATURE_NAME_MAP = {
    "crescent_prob": "Crescent-shaped changes (prob)",
    "fibrosis_prob": "Interstitial fibrosis (prob)",
    "crescent_logit_prob": "Crescent-shaped changes (logit prob)",
    "fibrosis_logit_prob": "Interstitial fibrosis (logit prob)",
    "crescent_median_prob": "Crescent-shaped changes (median prob)",
    "fibrosis_median_prob": "Interstitial fibrosis (median prob)",
    "crescent_std_prob": "Crescent-shaped changes (std prob)",
    "fibrosis_std_prob": "Interstitial fibrosis (std prob)",
    "crescent_min_prob": "Crescent-shaped changes (min prob)",
    "fibrosis_min_prob": "Interstitial fibrosis (min prob)",
    "crescent_max_prob": "Crescent-shaped changes (max prob)",
    "fibrosis_max_prob": "Interstitial fibrosis (max prob)",
    "crescent_pos_ratio": "Crescent-shaped changes (pos ratio)",
    "fibrosis_pos_ratio": "Interstitial fibrosis (pos ratio)",
    "crescent_entropy": "Crescent-shaped changes (entropy)",
    "fibrosis_entropy": "Interstitial fibrosis (entropy)",
    "crescent_pred": "Crescent-shaped changes (pred)",
    "fibrosis_pred": "Interstitial fibrosis (pred)",
    "ePWV": "ePWV",
    "SII": "SII",
    "24h-UP": "24h-UP",
    "eGFR": "eGFR",
    "Age": "Age",
    "Gender": "Gender",
}


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


def plot_global_importance(shap_values: np.ndarray, X: np.ndarray, feature_names: list[str]) -> None:
    import shap

    left_margin = 0.40
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    values = mean_abs[order]
    labels = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    y_pos = np.arange(len(labels))
    bar_color = "#E31A1C"
    ax.barh(y_pos, values, color=bar_color)
    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("mean(|SHAP| value)")
    ax.set_xlim(0, values.max() * 1.25)
    ax.grid(axis="x", alpha=0.2)

    offset = values.max() * 0.02
    for i, v in enumerate(values):
        ax.text(v + offset, i, f"+{v:.2f}", va="center", color=bar_color, fontsize=9)

    # Match beeswarm layout so labels are not clipped
    fig.subplots_adjust(left=left_margin, right=0.95, top=0.95, bottom=0.12)
    plt.savefig(FIGURES_DIR / "shap_global.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.8, 4.5))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        color=plt.get_cmap("coolwarm"),
    )
    fig = plt.gcf()
    axes = fig.axes
    if axes:
        # Keep layout aligned with bar plot and leave room for labels
        axes[0].set_position([left_margin, 0.12, 0.45, 0.78])
    if len(axes) > 1:
        axes[1].set_position([0.88, 0.12, 0.03, 0.78])
    plt.savefig(FIGURES_DIR / "shap_beeswarm.png", dpi=300)
    plt.close()


def plot_force(shap_values: np.ndarray, base_value: float, feature_names: list[str], sample: np.ndarray, out_name: str) -> None:
    import shap

    plt.figure(figsize=(7.8, 2.4))
    shap.force_plot(
        base_value,
        shap_values,
        sample,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    ax = plt.gca()
    # Shift header texts (higher/arrow/lower/f(x)/value) upward together
    for text in ax.texts:
        x, y = text.get_position()
        if y > 0.2:
            text.set_position((x, min(0.98, y + 0.14)))

    # Nudge higher/lower/arrows a bit further upward
    for text in ax.texts:
        label = text.get_text().strip().lower()
        if label in {"higher", "lower", "$\\leftarrow$", "$\\rightarrow$"}:
            x, y = text.get_position()
            text.set_position((x, min(0.98, y + 0.02)))

    # Nudge numeric value labels slightly downward to avoid overlapping f(x) text
    for text in ax.texts:
        label = text.get_text().strip()
        try:
            float(label)
        except ValueError:
            continue
        x, y = text.get_position()
        if y > 0.2:
            text.set_position((x, max(0.2, y - 0.05)))
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / out_name, dpi=300)
    plt.close()


def main() -> None:
    setup_matplotlib()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data = build_dataset()

    with open(METRICS_DIR / "rf_stage_summary.json", "r", encoding="utf-8") as f:
        stage_summary = json.load(f)
    stage_features = stage_summary.get("features", [])
    display_features = [FEATURE_NAME_MAP.get(f, f) for f in stage_features]

    data = data.dropna(subset=stage_features + ["1yearegfr"])
    val_df = data[data["split"] == "val"].copy()
    X_val = val_df[stage_features].values

    model = joblib.load(MODELS_DIR / "rf_stage_model.joblib")

    try:
        import shap
    except Exception as exc:
        raise RuntimeError("未安装 shap，请先安装后再运行 SHAP 分析。") from exc

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1]

    plot_global_importance(shap_values, X_val, display_features)
    # 选择两个样本用于力图展示
    if len(X_val) > 0:
        plot_force(shap_values[0], base_value, display_features, X_val[0], "shap_force_1.png")
    if len(X_val) > 1:
        plot_force(shap_values[1], base_value, display_features, X_val[1], "shap_force_2.png")

    summary = {
        "samples": int(len(X_val)),
        "features": stage_features,
        "global_plot": str((FIGURES_DIR / "shap_global.png").resolve()),
        "beeswarm_plot": str((FIGURES_DIR / "shap_beeswarm.png").resolve()),
        "force_plot_1": str((FIGURES_DIR / "shap_force_1.png").resolve()),
        "force_plot_2": str((FIGURES_DIR / "shap_force_2.png").resolve()),
    }
    with open(METRICS_DIR / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("SHAP 分析完成：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
