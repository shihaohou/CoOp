"""Stitch gradient-inversion snapshots into a Fig. 7-style figure.

Inputs are two (or more) output directories, one per row, each produced
by ``privacy_attack_dlg.py`` or ``privacy_attack_coop.py``. Each
directory must contain ``original.png`` plus ``iter_XYZ.png`` for the
iterations listed in ``--iters``.

Because the full-row attack runs at native 32x32 (CIFAR-10) while the
prompt-row attack runs at CLIP's 224x224, the raw PNGs from the two
rows have different resolutions. This script upsamples everything to a
common pixel size with nearest-neighbor interpolation before placing
it on the grid, so both rows occupy the same visual area in the
stitched PDF (nearest-neighbor preserves the chunky pixel look of the
32x32 row instead of bicubic-blurring it).
"""

import argparse
import os
from typing import List

# Force a non-interactive backend so the script works on headless servers
# regardless of matplotlibrc.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def _load_png(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _row_images(row_dir: str, iters: List[int]) -> List[Image.Image]:
    images = [_load_png(os.path.join(row_dir, "original.png"))]
    for it in iters:
        images.append(_load_png(os.path.join(row_dir, f"iter_{it:03d}.png")))
    return images


def _resize_to(img: Image.Image, target: int) -> Image.Image:
    if img.size == (target, target):
        return img
    return img.resize((target, target), Image.NEAREST)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row-dirs", nargs="+", required=True,
                   help="one output dir per row, in top-to-bottom order")
    p.add_argument("--row-labels", nargs="+", required=True,
                   help="left-side label for each row; must match --row-dirs length")
    p.add_argument("--iters", default="0,20,40,60,80,100")
    p.add_argument("--output", required=True,
                   help="output figure path (PNG or PDF); a sibling .png is "
                        "always saved as well for quick inspection")
    p.add_argument("--target-size", type=int, default=0,
                   help="resize every cell to this many pixels per side with "
                        "nearest-neighbor; 0 = use max size across all inputs")
    p.add_argument("--cell-size", type=float, default=1.6,
                   help="approximate inches per cell")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    if len(args.row_dirs) != len(args.row_labels):
        raise ValueError("--row-dirs and --row-labels must have equal length")

    iters = [int(x) for x in args.iters.split(",") if x.strip()]
    n_rows = len(args.row_dirs)
    n_cols = 1 + len(iters)

    # Load everything first, then pick a common pixel size.
    rows_imgs: List[List[Image.Image]] = []
    for row_dir in args.row_dirs:
        imgs = _row_images(row_dir, iters)
        sizes = [im.size for im in imgs]
        print(f"loaded {row_dir}: {sizes[0]} + {len(iters)} iters at {sizes[1]}")
        rows_imgs.append(imgs)

    if args.target_size > 0:
        target = args.target_size
    else:
        target = max(max(im.size) for row in rows_imgs for im in row)
    print(f"target cell pixel size = {target}")

    rows_imgs = [[_resize_to(im, target) for im in row] for row in rows_imgs]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(args.cell_size * n_cols, args.cell_size * n_rows + 0.4),
        squeeze=False,
    )

    col_titles = ["Original"] + [f"Iter {it}" for it in iters]

    for r, (imgs, row_label) in enumerate(zip(rows_imgs, args.row_labels)):
        for c in range(n_cols):
            ax = axes[r, c]
            # numpy.asarray makes PIL -> matplotlib path explicit, avoids
            # surprises with some PIL/matplotlib version combos.
            ax.imshow(np.asarray(imgs[c]))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11)

    fig.subplots_adjust(
        left=0.05, right=0.98, top=0.92, bottom=0.05,
        wspace=0.05, hspace=0.05,
    )

    fig.savefig(args.output, dpi=args.dpi)
    out_size = os.path.getsize(args.output)
    print(f"Saved {args.output} ({out_size:,} bytes)")

    # Also save a PNG sibling so the user can sanity-check without a PDF viewer.
    base, ext = os.path.splitext(args.output)
    if ext.lower() != ".png":
        sibling = base + ".png"
        fig.savefig(sibling, dpi=args.dpi)
        print(f"Saved {sibling} ({os.path.getsize(sibling):,} bytes)")

    plt.close(fig)


if __name__ == "__main__":
    main()
