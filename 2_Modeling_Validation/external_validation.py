# -*- coding: utf-8 -*-
"""
外部验证完整流程：
Excel 临床数据 + 外部验证图像 + 两个 ResNet18 图像模型 ->
生成 crescent_prob / fibrosis_prob / crescent_max_prob / fibrosis_max_prob ->
合并外部验证表 -> 输入三个随机森林模型 -> 画同一张 ROC 曲线。

使用方法：
1）把本文件放到项目目录，或单独放到任意目录。
2）修改下面【用户配置区】里的路径。
3）运行：python external_validation_3rf_full.py

依赖：pandas numpy matplotlib scikit-learn joblib torch torchvision pillow openpyxl tqdm
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import roc_auc_score, roc_curve

ImageFile.LOAD_TRUNCATED_IMAGES = True


# =============================================================================
# 1. 用户配置区：请改成你电脑上的真实路径
# =============================================================================

# 外部验证 Excel。你上传的表格 sheet 名是“外部验证”。
EXCEL_PATH = r"C:\Users\Olympus\Desktop\验证\验证_Excel.xlsx"
SHEET_NAME = "外部验证"

# 外部验证图像文件夹。
# 文件名需要能解析出病理号，例如：K110_1.tif、K110-1.jpg、K110.png。
CRESCENT_IMAGE_DIR = r"C:\Users\Olympus\Desktop\验证\验证_Crescent"
FIBROSIS_IMAGE_DIR = r"C:\Users\Olympus\Desktop\验证\验证_Fiberosis"

# 两个 ResNet18 图像模型。
CRESCENT_RESNET_PATH = r"C:\Users\Olympus\Desktop\验证\models\crescent_resnet18_best.pt"
FIBROSIS_RESNET_PATH = r"C:\Users\Olympus\Desktop\验证\models\fibrosis_resnet18_best.pt"

# 三个随机森林模型。
RF_MODEL_PATHS = {
    "Clinical model": r"C:\Users\Olympus\Desktop\验证\models\rf_clinical_model.joblib",
    "Fusion model":   r"C:\Users\Olympus\Desktop\验证\models\rf_fusion_model.joblib",
    "Stage model":    r"C:\Users\Olympus\Desktop\验证\models\rf_stage_model.joblib",
}

# 输出目录。
OUTPUT_DIR = r"C:\Users\Olympus\Desktop\验证\external_validation_outputs_3rf"

# Excel 关键列名。你的验证表当前包含：1yearegfr、病理号、ePWV、SII、24h-UP、eGFR。
ID_COL = "病理号"
LABEL_COL = "1yearegfr"
CLINICAL_FEATURES = ["ePWV", "SII", "24h-UP", "eGFR"]

# 三个 RF 的输入变量顺序必须和训练时一致。
# 结合你上传的训练脚本：
# Clinical = 4 个临床变量；Fusion = 两个 mean prob + 临床变量；Stage = 两个 max prob + 临床变量。
RF_FEATURES = {
    "Clinical model": ["ePWV", "SII", "24h-UP", "eGFR"],
    "Fusion model": ["crescent_prob", "fibrosis_prob", "ePWV", "SII", "24h-UP", "eGFR"],
    "Stage model": ["crescent_max_prob", "fibrosis_max_prob", "ePWV", "SII", "24h-UP", "eGFR"],
}

# 图像预测参数。
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0       # Windows 上建议先用 0，稳定后可改为 2 或 4。
BOOTSTRAP_REPEATS = 2000
RANDOM_SEED = 42

IMAGE_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}


# =============================================================================
# 2. 通用工具函数
# =============================================================================

def normalize_id(x) -> str:
    """统一病理号格式，避免 Excel 数字、空格、大小写导致匹配失败。"""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.upper()


def extract_patient_id_from_filename(path: str) -> str:
    """
    从图像文件名提取病理号。
    默认规则：取第一个下划线或连字符前面的部分。
    例如：K110_1.tif -> K110；K110-1.jpg -> K110；K110.tif -> K110。
    如果你的文件名规则不同，请修改这里。
    """
    stem = Path(path).stem
    pid = re.split(r"[_\-]", stem)[0]
    return normalize_id(pid)


def list_images_by_patient(image_dir: str) -> Dict[str, List[str]]:
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"图像文件夹不存在：{image_dir}")

    mapping: Dict[str, List[str]] = {}
    for p in image_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            pid = extract_patient_id_from_filename(str(p))
            mapping.setdefault(pid, []).append(str(p))

    for pid in mapping:
        mapping[pid] = sorted(mapping[pid])
    return mapping


def ensure_file(path: str, name: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{name}不存在：{p}")
    return p


def ensure_dir(path: str, name: str) -> Path:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"{name}不存在：{p}")
    return p


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =============================================================================
# 3. ResNet18 模型自适应加载
# =============================================================================

class WrappedResNet18(nn.Module):
    def __init__(self, fc_type: str, hidden_dim: int = 512, dropout_p: float = 0.5):
        super().__init__()
        self.backbone = build_resnet18_backbone(fc_type, hidden_dim=hidden_dim, dropout_p=dropout_p)

    def forward(self, x):
        return self.backbone(x)


def build_resnet18_backbone(fc_type: str, hidden_dim: int = 512, dropout_p: float = 0.5) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    if fc_type == "simple":
        model.fc = nn.Linear(in_features, 2)
    elif fc_type == "two_layer":
        # 注意：你的 checkpoint 里 fc.0.weight 是 [128, 512]，
        # 说明训练时分类头隐藏层是 128，不是 512。
        # 这里用 hidden_dim 自动匹配 checkpoint，避免 size mismatch。
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 2),
        )
    else:
        raise ValueError(f"未知 fc_type: {fc_type}")
    return model


def _strip_prefix_if_present(state: dict, prefix: str) -> dict:
    if all(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def load_resnet18_binary(model_path: str, device: torch.device) -> nn.Module:
    """
    自动兼容常见保存格式：
    1）state_dict 直接保存；
    2）包含 model_state_dict / state_dict 的 checkpoint；
    3）key 带 module. 前缀；
    4）模型 key 是 backbone.xxx 或直接 xxx；
    5）fc 是简单 Linear 或两层全连接头。
    """
    raw = torch.load(model_path, map_location=device)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        state = raw["model_state_dict"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        state = raw["state_dict"]
    elif isinstance(raw, dict):
        state = raw
    else:
        raise TypeError(f"不支持的 ResNet 权重格式：{type(raw)}")

    state = {k.replace("module.", ""): v for k, v in state.items()}
    keys = list(state.keys())

    wrapped = any(k.startswith("backbone.") for k in keys)
    check_keys = [k.replace("backbone.", "") for k in keys]

    hidden_dim = 512
    if any(k.startswith("fc.0.") for k in check_keys):
        fc_type = "two_layer"
        # 自动读取两层分类头的隐藏层维度。
        # 例如 fc.0.weight shape = [128, 512]，则 hidden_dim = 128。
        fc0_weight_key = None
        for k in keys:
            kk = k.replace("backbone.", "")
            if kk == "fc.0.weight":
                fc0_weight_key = k
                break
        if fc0_weight_key is not None:
            hidden_dim = int(state[fc0_weight_key].shape[0])
    elif any(k.startswith("fc.") for k in check_keys):
        fc_type = "simple"
    else:
        raise ValueError(
            f"无法从权重文件判断 ResNet18 分类头结构：{model_path}\n"
            f"前几个 key：{keys[:10]}"
        )

    model = WrappedResNet18(fc_type, hidden_dim=hidden_dim) if wrapped else build_resnet18_backbone(fc_type, hidden_dim=hidden_dim)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] 加载 {Path(model_path).name} 时 missing keys 前 10 个：{missing[:10]}")
    if unexpected:
        print(f"[WARN] 加载 {Path(model_path).name} 时 unexpected keys 前 10 个：{unexpected[:10]}")

    model.to(device)
    model.eval()
    if fc_type == "two_layer":
        print(f"[INFO] 已加载 {Path(model_path).name}，结构：{'Wrapped' if wrapped else 'Direct'} ResNet18，分类头：{fc_type}，hidden_dim={hidden_dim}")
    else:
        print(f"[INFO] 已加载 {Path(model_path).name}，结构：{'Wrapped' if wrapped else 'Direct'} ResNet18，分类头：{fc_type}")
    return model


def validation_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: List[str]):
        self.image_paths = image_paths
        self.tfm = validation_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.tfm(img), path


@torch.no_grad()
def predict_images(model: nn.Module, image_paths: List[str], device: torch.device) -> pd.DataFrame:
    if len(image_paths) == 0:
        return pd.DataFrame(columns=["image_path", "patient_id", "prob"])

    ds = ImagePathDataset(image_paths)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    rows = []
    for x, paths in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        for p, prob in zip(paths, probs):
            rows.append({
                "image_path": p,
                "patient_id": extract_patient_id_from_filename(p),
                "prob": float(prob),
            })
    return pd.DataFrame(rows)


def aggregate_patient_predictions(pred_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if pred_df.empty:
        return pd.DataFrame(columns=["patient_id", f"{prefix}_prob", f"{prefix}_max_prob", f"n_{prefix}_images"])

    agg = pred_df.groupby("patient_id")["prob"].agg(
        **{
            f"{prefix}_prob": "mean",
            f"{prefix}_max_prob": "max",
            f"n_{prefix}_images": "count",
        }
    ).reset_index()
    return agg


# =============================================================================
# 4. Excel 读取、RF 预测和 ROC
# =============================================================================

def read_external_excel() -> pd.DataFrame:
    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    required = [ID_COL, LABEL_COL] + CLINICAL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Excel 缺少必要列：{missing}\n当前列名：{list(df.columns)}")

    df = df.copy()
    df["patient_id"] = df[ID_COL].map(normalize_id)
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def predict_rf_positive_prob(model, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("随机森林模型不支持 predict_proba，无法画 ROC。")
    prob = model.predict_proba(X)
    if prob.ndim != 2 or prob.shape[1] != 2:
        raise ValueError(f"predict_proba 输出不是二分类概率，shape={prob.shape}")
    return prob[:, 1]


def bootstrap_auc_ci(y_true, y_prob, n_boot=2000, seed=42) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    auc = float(roc_auc_score(y_true, y_prob))
    if len(aucs) == 0:
        return auc, np.nan, np.nan
    return auc, float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def safe_load_joblib(path: str, model_name: str):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        msg = str(e)
        raise RuntimeError(
            f"加载 {model_name} 失败：{e}\n\n"
            "这通常是 joblib 模型保存环境和当前环境版本不一致导致。\n"
            "你之前报过 `No module named numpy._core`，一般需要把当前环境升级到 numpy 2.x，\n"
            "并尽量使用训练模型时相同的 scikit-learn 版本。可先尝试：\n"
            "  pip install -U numpy scikit-learn joblib\n"
            "如果仍不行，请在原训练环境里重新保存模型，或用原训练环境运行本脚本。"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"加载 {model_name} 失败：{e}\n"
            "请确认 scikit-learn / numpy / joblib 版本与训练保存模型时一致。"
        ) from e


def main() -> None:
    warnings.filterwarnings("default")

    ensure_file(EXCEL_PATH, "外部验证 Excel")
    ensure_dir(CRESCENT_IMAGE_DIR, "Crescent 图像文件夹")
    ensure_dir(FIBROSIS_IMAGE_DIR, "Fibrosis 图像文件夹")
    ensure_file(CRESCENT_RESNET_PATH, "crescent_resnet18_best.pt")
    ensure_file(FIBROSIS_RESNET_PATH, "fibrosis_resnet18_best.pt")
    for name, path in RF_MODEL_PATHS.items():
        ensure_file(path, name)

    outdir = Path(OUTPUT_DIR)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] 读取外部验证 Excel...")
    external_df = read_external_excel()
    print(f"[INFO] 外部验证 Excel 病例数：{len(external_df)}")

    print("[INFO] 扫描外部验证图像...")
    crescent_map = list_images_by_patient(CRESCENT_IMAGE_DIR)
    fibrosis_map = list_images_by_patient(FIBROSIS_IMAGE_DIR)
    crescent_paths = [p for paths in crescent_map.values() for p in paths]
    fibrosis_paths = [p for paths in fibrosis_map.values() for p in paths]
    print(f"[INFO] Crescent 图像数：{len(crescent_paths)}，病例数：{len(crescent_map)}")
    print(f"[INFO] Fibrosis 图像数：{len(fibrosis_paths)}，病例数：{len(fibrosis_map)}")

    device = get_device()
    print(f"[INFO] 使用设备：{device}")

    print("[INFO] 加载并预测 Crescent ResNet18...")
    crescent_model = load_resnet18_binary(CRESCENT_RESNET_PATH, device)
    crescent_img_pred = predict_images(crescent_model, crescent_paths, device)
    crescent_patient = aggregate_patient_predictions(crescent_img_pred, "crescent")

    print("[INFO] 加载并预测 Fibrosis ResNet18...")
    fibrosis_model = load_resnet18_binary(FIBROSIS_RESNET_PATH, device)
    fibrosis_img_pred = predict_images(fibrosis_model, fibrosis_paths, device)
    fibrosis_patient = aggregate_patient_predictions(fibrosis_img_pred, "fibrosis")

    crescent_img_pred.to_csv(outdir / "crescent_image_predictions.csv", index=False, encoding="utf-8-sig")
    fibrosis_img_pred.to_csv(outdir / "fibrosis_image_predictions.csv", index=False, encoding="utf-8-sig")

    image_patient = pd.merge(crescent_patient, fibrosis_patient, on="patient_id", how="outer")
    image_patient.to_csv(outdir / "patient_image_predictions.csv", index=False, encoding="utf-8-sig")

    print("[INFO] 合并图像预测到外部验证表...")
    merged = external_df.merge(image_patient, on="patient_id", how="left")
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged.to_csv(outdir / "external_validation_merged_with_image_probs.csv", index=False, encoding="utf-8-sig")

    missing_crescent = merged["crescent_prob"].isna().sum() if "crescent_prob" in merged.columns else len(merged)
    missing_fibrosis = merged["fibrosis_prob"].isna().sum() if "fibrosis_prob" in merged.columns else len(merged)
    print(f"[WARN] 未匹配到 Crescent 图像预测的病例数：{missing_crescent}")
    print(f"[WARN] 未匹配到 Fibrosis 图像预测的病例数：{missing_fibrosis}")

    all_pred = merged[["patient_id", ID_COL, LABEL_COL]].copy()
    auc_rows = []

    plt.figure(figsize=(8, 6), dpi=200)

    for model_name, model_path in RF_MODEL_PATHS.items():
        print(f"\n[INFO] 正在外部验证：{model_name}")
        model = safe_load_joblib(model_path, model_name)
        features = RF_FEATURES[model_name]

        missing_cols = [c for c in features if c not in merged.columns]
        if missing_cols:
            raise ValueError(f"{model_name} 缺少输入变量：{missing_cols}")

        model_df = merged[["patient_id", ID_COL, LABEL_COL] + features].copy()
        before = len(model_df)
        model_df = model_df.dropna(subset=[LABEL_COL] + features)
        after = len(model_df)
        if after < before:
            print(f"[WARN] {model_name} 因标签或输入变量缺失删除 {before - after} 例，剩余 {after} 例。")
        if after == 0:
            raise ValueError(f"{model_name} 没有可验证样本。")
        if model_df[LABEL_COL].nunique() < 2:
            raise ValueError(f"{model_name} 的验证样本只有一个类别，无法计算 ROC/AUC。")

        y_true = model_df[LABEL_COL].astype(int).values
        # 训练脚本里 RF 使用 .values 训练，因此这里也使用 numpy 数组，严格保持变量顺序。
        X = model_df[features].values
        y_prob = predict_rf_positive_prob(model, X)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc, ci_low, ci_high = bootstrap_auc_ci(y_true, y_prob, BOOTSTRAP_REPEATS, RANDOM_SEED)

        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC = {auc:.3f})")

        prob_col = model_name.replace(" ", "_").lower() + "_prob"
        temp = model_df[["patient_id"]].copy()
        temp[prob_col] = y_prob
        all_pred = all_pred.merge(temp, on="patient_id", how="left")

        auc_rows.append({
            "Model": model_name,
            "N": int(after),
            "Positive": int(y_true.sum()),
            "Negative": int((1 - y_true).sum()),
            "AUC": auc,
            "AUC_95CI_lower": ci_low,
            "AUC_95CI_upper": ci_high,
            "Features": ", ".join(features),
        })

        print(f"[RESULT] {model_name}: AUC = {auc:.3f}, 95% CI = {ci_low:.3f}-{ci_high:.3f}, n = {after}")
        print(f"[RESULT] 输入变量：{features}")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Chance")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("External Validation ROC Curves", fontsize=15)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    roc_path = figdir / "external_validation_3rf_ROC.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    auc_df = pd.DataFrame(auc_rows)
    auc_path = outdir / "external_validation_3rf_AUC_results.csv"
    pred_path = outdir / "external_validation_3rf_predictions.csv"
    auc_df.to_csv(auc_path, index=False, encoding="utf-8-sig")
    all_pred.to_csv(pred_path, index=False, encoding="utf-8-sig")

    summary = {
        "excel_path": EXCEL_PATH,
        "sheet_name": SHEET_NAME,
        "crescent_image_dir": CRESCENT_IMAGE_DIR,
        "fibrosis_image_dir": FIBROSIS_IMAGE_DIR,
        "output_dir": str(outdir.resolve()),
        "n_excel_cases": int(len(external_df)),
        "n_merged_cases": int(len(merged)),
        "missing_crescent_cases": int(missing_crescent),
        "missing_fibrosis_cases": int(missing_fibrosis),
        "rf_features": RF_FEATURES,
        "auc_results": auc_rows,
    }
    with open(outdir / "external_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[DONE] 三个随机森林模型外部验证完成。")
    print(f"[DONE] 合并后的外部验证表：{outdir / 'external_validation_merged_with_image_probs.csv'}")
    print(f"[DONE] 病例级图像预测：{outdir / 'patient_image_predictions.csv'}")
    print(f"[DONE] 三模型 ROC 图：{roc_path}")
    print(f"[DONE] AUC 结果：{auc_path}")
    print(f"[DONE] 三模型预测概率：{pred_path}")


if __name__ == "__main__":
    main()
