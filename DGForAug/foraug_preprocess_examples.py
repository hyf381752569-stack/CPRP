#!/usr/bin/env python3
"""
Single-image ForAug-style preprocessing.

Given an RGB image and its foreground mask, the script
creates an RGBA foreground cutout and an inpainted clean background.
By default the attentive eraser from the ForAug repository is used
to perform the background reconstruction.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
_foraug_candidates = [
    THIS_DIR / "ForAug-main",
    THIS_DIR.parent / "ForAug-main",
]
FORAUG_REPO = next((path for path in _foraug_candidates if path.exists()), None)
if FORAUG_REPO is None:
    raise FileNotFoundError("Could not locate ForAug-main repository. Checked: {}".format(_foraug_candidates))
if str(FORAUG_REPO) not in sys.path:
    sys.path.append(str(FORAUG_REPO))


def to_numpy_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def to_numpy_mask(mask: Image.Image) -> np.ndarray:
    mask_arr = np.array(mask.convert("L"))
    if mask_arr.max() <= 1:
        mask_arr = (mask_arr * 255).astype(np.uint8)
    return mask_arr


def create_foreground_rgba(image_np: np.ndarray, mask_np: np.ndarray) -> Image.Image:
    if mask_np.ndim != 2:
        raise ValueError("Mask must be single channel.")
    alpha = mask_np.astype(np.uint8)
    rgb = image_np.copy()
    rgb[alpha == 0] = 0
    rgba = np.dstack([rgb, alpha])
    return Image.fromarray(rgba, mode="RGBA")


@contextmanager
def temporarily_chdir(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class SingleImagePreprocessor:
    def __init__(self, inpaint_model: str) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if inpaint_model == "attentive_eraser" and self.device.type != "cuda":
            raise RuntimeError(
                "Attentive Eraser requires a CUDA-capable GPU. "
                "Please switch to a GPU environment or use the 'lama' inpaint model."
            )
        self.inpainter = self._init_inpainter(inpaint_model)

    def _init_inpainter(self, name: str):
        model_name = name.lower()
        try:
            if model_name == "attentive_eraser":
                from attentive_eraser import AttentiveEraser  # type: ignore

                try:
                    custom_root = FORAUG_REPO
                    if not custom_root.exists():
                        raise FileNotFoundError(f"ForAug repository not found at {custom_root}")
                    with temporarily_chdir(custom_root):
                        return AttentiveEraser(device=self.device)
                except Exception as exc:  # noqa: BLE001 - provide actionable message
                    raise RuntimeError(
                        "Failed to load the Attentive Eraser pipeline. "
                        "Ensure the required Hugging Face weights are available locally "
                        "(e.g. run with internet access once or download via "
                        "`huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 ...`) "
                        "or rerun with `--inpaint-model lama` / `INPAINT_MODEL=lama`."
                    ) from exc
            if model_name == "lama":
                from infill_lama import LaMa  # type: ignore

                return LaMa(device=self.device)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", str(exc))
            raise ModuleNotFoundError(
                f"Failed to import '{missing}'. Ensure the ForAug dependencies are available."
            ) from exc
        raise ValueError("Unsupported inpaint model. Choose from ['attentive_eraser', 'lama'].")

    def process(
        self,
        rgb_path: Path,
        mask_path: Path,
        output_dir: Path,
        prefix: str,
        background_format: str,
    ) -> Tuple[Path, Path]:
        rgb_image = Image.open(rgb_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")

        if mask_image.size != rgb_image.size:
            mask_image = mask_image.resize(rgb_image.size, Image.NEAREST)

        rgb_np = to_numpy_rgb(rgb_image)
        mask_np = to_numpy_mask(mask_image)

        foreground_rgba = create_foreground_rgba(rgb_np, mask_np)
        clean_background_raw = self.inpainter(rgb_np, mask_np)

        if isinstance(clean_background_raw, Image.Image):
            clean_background = clean_background_raw.convert("RGB")
        else:
            clean_background = Image.fromarray(np.asarray(clean_background_raw).astype(np.uint8)).convert("RGB")

        output_dir.mkdir(parents=True, exist_ok=True)
        fg_path = output_dir / f"{prefix}_foreground.png"
        bg_path = output_dir / f"{prefix}_background.{background_format}"

        foreground_rgba.save(fg_path)
        if background_format == "jpg":
            clean_background.save(bg_path, quality=95)
        else:
            clean_background.save(bg_path)

        return fg_path, bg_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ForAug-style foreground/background pair for a single image.")
    parser.add_argument("--input-rgb", type=Path, required=True, help="Path to the RGB image.")
    parser.add_argument("--input-mask", type=Path, required=True, help="Path to the foreground mask.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store the outputs.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Filename prefix for outputs. Defaults to the RGB filename stem.",
    )
    parser.add_argument(
        "--inpaint-model",
        type=str,
        default="attentive_eraser",
        choices=["attentive_eraser", "lama"],
        help="Inpainting model to use.",
    )
    parser.add_argument(
        "--background-format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Image format for the reconstructed background.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgb_path = args.input_rgb.expanduser().resolve()
    mask_path = args.input_mask.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask image not found: {mask_path}")

    prefix = args.output_prefix or rgb_path.stem
    processor = SingleImagePreprocessor(args.inpaint_model)
    fg_path, bg_path = processor.process(rgb_path, mask_path, output_dir, prefix, args.background_format)

    print("[Info] Foreground saved to:", fg_path)
    print("[Info] Background saved to:", bg_path)


if __name__ == "__main__":
    main()
