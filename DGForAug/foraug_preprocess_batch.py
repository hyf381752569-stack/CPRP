#!/usr/bin/env python3
"""
Batch wrapper around SingleImagePreprocessor that avoids reloading the diffusion
pipeline for each image.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

from tqdm import tqdm

from foraug_preprocess_examples import SingleImagePreprocessor


def find_existing_file(directory: Path, stem: str, exts: Tuple[str, ...]) -> Optional[Path]:
    for ext in exts:
        candidate = directory / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_pairs(rgb_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for rgb_path in sorted(rgb_dir.iterdir()):
        if not rgb_path.is_file():
            continue
        stem = rgb_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if not mask_path.exists():
            mask_path = mask_dir / f"{stem}.jpg"
        if not mask_path.exists():
            mask_path = mask_dir / f"{stem}.jpeg"
        if not mask_path.exists():
            tqdm.write(f"[Warn] Mask missing for {rgb_path.name}, skipping.")
            continue
        pairs.append((rgb_path, mask_path))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate ForAug-style foreground/background assets.")
    parser.add_argument("--input-rgb-dir", type=Path, required=True, help="Directory containing RGB images.")
    parser.add_argument("--input-mask-dir", type=Path, required=True, help="Directory containing masks.")
    parser.add_argument("--foreground-dir", type=Path, required=True, help="Output directory for RGBA foregrounds.")
    parser.add_argument("--background-dir", type=Path, required=True, help="Output directory for backgrounds.")
    parser.add_argument(
        "--inpaint-model",
        type=str,
        default="attentive_eraser",
        choices=["attentive_eraser", "lama"],
        help="Which inpainting model to use.",
    )
    parser.add_argument(
        "--background-format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="File format for background images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rgb_dir = args.input_rgb_dir.expanduser().resolve()
    mask_dir = args.input_mask_dir.expanduser().resolve()
    fg_dir = args.foreground_dir.expanduser().resolve()
    bg_dir = args.background_dir.expanduser().resolve()

    for path in (rgb_dir, mask_dir):
        if not path.exists():
            raise FileNotFoundError(f"Input directory not found: {path}")

    fg_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(rgb_dir, mask_dir)
    if not pairs:
        print("[Info] No matching RGB/Mask pairs found.")
        return

    processor = SingleImagePreprocessor(args.inpaint_model)

    processed = 0
    failed = 0
    skipped_existing = 0

    with tqdm(pairs, desc="Inpainting backgrounds", unit="img") as iterator:
        for rgb_path, mask_path in iterator:
            stem = rgb_path.stem
            existing_bg = find_existing_file(
                bg_dir,
                stem,
                exts=(f".{args.background_format}", ".jpg", ".jpeg", ".png", ".webp"),
            )
            existing_fg = find_existing_file(fg_dir, stem, exts=(".png", ".webp"))
            if existing_bg is not None and existing_fg is not None:
                skipped_existing += 1
                tqdm.write(f"[Skip] Assets already exist: {existing_fg.name}, {existing_bg.name}")
                continue

            target_fg = fg_dir / f"{stem}.png"
            target_bg = bg_dir / f"{stem}.{args.background_format}"

            try:
                tmp_dir = Path(tempfile.mkdtemp(prefix="foraug_pre_"))
                fg_path, bg_path = processor.process(
                    rgb_path=rgb_path,
                    mask_path=mask_path,
                    output_dir=tmp_dir,
                    prefix=stem,
                    background_format=args.background_format,
                )

                shutil.move(str(fg_path), target_fg)
                shutil.move(str(bg_path), target_bg)
                processed += 1
            except Exception as exc:
                failed += 1
                tqdm.write(f"[Error] Failed on {rgb_path.name}: {exc}")
            finally:
                if 'tmp_dir' in locals() and tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[Info] Completed. Processed {processed} images, failed {failed}, skipped existing {skipped_existing}.")
    print(f"[Info] Foregrounds: {fg_dir}")
    print(f"[Info] Backgrounds: {bg_dir}")


if __name__ == "__main__":
    main()
