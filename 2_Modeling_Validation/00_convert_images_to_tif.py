from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.utils.config import IMAGE_DIRS_RAW, IMAGE_DIRS_TIF, IMAGE_TIF_ROOT
from src.utils.io import is_image_file


def _output_name(src: Path) -> str:
    ext = src.suffix.lower().lstrip(".")
    stem = src.stem
    if ext in {"tif", "tiff"}:
        return f"{stem}.tif"
    return f"{stem}__{ext}.tif"


def _convert_image(src: Path, dst: Path) -> None:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.save(dst, format="TIFF", compression="tiff_deflate")


def _is_valid_tif(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def _prepare_dir(src_dir: Path, dst_dir: Path) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_dir": str(src_dir),
        "target_dir": str(dst_dir),
        "total": 0,
        "converted": 0,
        "copied": 0,
        "skipped": 0,
        "failed": 0,
    }
    for path in src_dir.rglob("*"):
        if not path.is_file() or not is_image_file(path):
            continue
        summary["total"] += 1
        out_name = _output_name(path)
        out_path = dst_dir / out_name
        if out_path.exists() and out_path.stat().st_size > 0:
            if _is_valid_tif(out_path):
                summary["skipped"] += 1
                continue
            out_path.unlink()
        try:
            if path.suffix.lower() in {".tif", ".tiff"}:
                shutil.copy2(path, out_path)
                summary["copied"] += 1
            else:
                _convert_image(path, out_path)
                summary["converted"] += 1
        except Exception:
            summary["failed"] += 1
    return summary


def main() -> None:
    IMAGE_TIF_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for key, src_dir in IMAGE_DIRS_RAW.items():
        dst_dir = IMAGE_DIRS_TIF[key]
        summaries[key] = _prepare_dir(src_dir, dst_dir)

    summary_path = IMAGE_TIF_ROOT / "tif_conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print("图像格式转换完成：")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"转换摘要已保存：{summary_path}")


if __name__ == "__main__":
    main()
