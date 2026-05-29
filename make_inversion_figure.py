"""Stitch gradient-inversion snapshots into a Fig. 7-style figure.

Inputs are two output directories (one per row), each produced by
``privacy_attack_coop.py``. Each directory must contain
``original.png`` plus ``iter_XYZ.png`` for the iterations listed in
``--iters``.

The default layout is the same as in the PromptFL gradient-inversion
figure: two rows, columns ``Original | Iter 0 | 20 | 40 | 60 | 80 | 100``,
with a horizontal divider between the two rows.
"""

import argparse
import os
from typing import List

import matplotlib.pyplot as plt
from PIL import Image


def _load_png(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _row_images(row_dir: str, iters: List[int]) -> List[Image.Image]:
    images = [_load_png(os.path.join(row_dir, "original.png"))]
    for it in iters:
        images.append(_load_png(os.path.join(row_dir, f"iter_{it:03d}.png")))
    return images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row-dirs", nargs="+", required=True,
                   help="one output dir per row, in top-to-bottom order")
    p.add_argument("--row-labels", nargs="+", required=True,
                   help="left-side label for each row; must match --row-dirs length")
    p.add_argument("--iters", default="0,20,40,60,80,100")
    p.add_argument("--output", required=True,
                   help="output figure path (PNG or PDF)")
    p.add_argument("--cell-size", type=float, default=1.6,
                   help="approximate inches per cell")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    if len(args.row_dirs) != len(args.row_labels):
        raise ValueError("--row-dirs and --row-labels must have equal length")

    iters = [int(x) for x in args.iters.split(",") if x.strip()]
    n_rows = len(args.row_dirs)
    n_cols = 1 + len(iters)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(args.cell_size * n_cols, args.cell_size * n_rows),
        squeeze=False,
    )

    col_titles = ["Original"] + [f"Iter {it}" for it in iters]

    for r, (row_dir, row_label) in enumerate(zip(args.row_dirs, args.row_labels)):
        imgs = _row_images(row_dir, iters)
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(imgs[c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=12)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11)

    # Vertical divider between Original column and Iter columns: draw a
    # thin line by tweaking the gridspec wspace and adding a line patch.
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
