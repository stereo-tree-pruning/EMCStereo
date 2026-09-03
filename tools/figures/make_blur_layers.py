"""Generate 3 progressively mosaic (pixelated block) versions of a disparity map.

Illustrates the coarse-to-fine Soft-Argmin outputs (disp1/disp2/disp3 in
src/disp_processor.py). Label 1 is the coarsest (biggest mosaic blocks), label 3
is the finest (smallest blocks). Edit --blocks to change the block size per layer.
"""

import argparse
from pathlib import Path

import cv2

# EMCStereo/  ->  repo root that holds the ground-truth image
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent / "DEFOM-Stereo_3305.png"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", type=Path, default=DEFAULT_INPUT,
        help="path to the ground-truth image",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="output folder (default: same folder as the input image)",
    )
    parser.add_argument(
        "-b", "--blocks", type=int, nargs=3, default=[96, 56, 32],
        metavar=("B1", "B2", "B3"),
        help="mosaic block size in pixels for layers 1, 2, 3 (coarse -> fine)",
    )
    return parser.parse_args()


def mosaic(image, block):
    """Pixelate `image` into square blocks of `block` pixels (hard edges)."""
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def main():
    args = parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")

    out_dir = args.output_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    for label, block in enumerate(args.blocks, start=1):
        out_path = out_dir / f"{stem}_{label}.png"
        cv2.imwrite(str(out_path), mosaic(image, block))
        print(f"[layer {label}] block={block:<4} -> {out_path}")


if __name__ == "__main__":
    main()
