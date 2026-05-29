"""Stitch gradient-inversion snapshots into a Fig. 7-style figure.

Inputs are two (or more) output directories, one per row, each produced
by ``privacy_attack_dlg.py`` or ``privacy_attack_coop.py``. Each
directory must contain ``original.png`` plus ``iter_XYZ.png`` for the
iterations listed in ``--iters``.

Output figure layout (matches the PromptFL Fig. 7 reference):

    | Original | <- vertical line -> | Iter 0 | Iter 20 | ... | Iter 100 |
    -------------------------- dashed horizontal divider -------------------
    | Original |                     | Iter 0 | Iter 20 | ... | Iter 100 |

  * cells have no borders
  * Original column is separated from the Iter columns by a solid line
  * adjacent rows are separated by a dashed line
  * everything is resized to a common pixel size (default: max across
    inputs; pass --target-size 32 to shrink the prompt row down to
    the full row's 32x32 native resolution)

With --save-resized, each post-resize cell is also written separately
to <row_dir>/resized_<target>_<resample>/ for use outside the figure.
"""

import argparse
import os
from typing import List

import matplotlib
matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402


_RESAMPLE_MAP = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def _load_png(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _row_images(row_dir: str, iters: List[int]) -> List[Image.Image]:
    images = [_load_png(os.path.join(row_dir, "original.png"))]
    for it in iters:
        images.append(_load_png(os.path.join(row_dir, f"iter_{it:03d}.png")))
    return images


def _resize_to(img: Image.Image, target: int, mode: str) -> Image.Image:
    if img.size == (target, target):
        return img
    if mode == "auto":
        src = max(img.size)
        method = Image.NEAREST if target >= src else Image.LANCZOS
    else:
        method = _RESAMPLE_MAP[mode]
    return img.resize((target, target), method)


def _add_vertical_divider(fig, axes, between: int = 0) -> None:
    """Solid line between column `between` and `between + 1`, spanning
    from below the bottom row to above the top row."""
    pos_l = axes[0, between].get_position()
    pos_r = axes[0, between + 1].get_position()
    x = (pos_l.x1 + pos_r.x0) / 2.0
    y_top = axes[0, between].get_position().y1 + 0.01
    y_bot = axes[-1, between].get_position().y0 - 0.01
    fig.add_artist(Line2D(
        [x, x], [y_bot, y_top],
        color="black", linewidth=1.4,
        transform=fig.transFigure,
    ))


def _add_horizontal_dashed_divider(fig, axes, between: int = 0) -> None:
    """Dashed line midway between rows `between` and `between + 1`,
    spanning the full figure width."""
    pos_top = axes[between, 0].get_position()
    pos_bot = axes[between + 1, 0].get_position()
    y = (pos_top.y0 + pos_bot.y1) / 2.0
    x_l = axes[between, 0].get_position().x0 - 0.02
    x_r = axes[between, -1].get_position().x1 + 0.02
    fig.add_artist(Line2D(
        [x_l, x_r], [y, y],
        color="black", linewidth=1.0, linestyle="--",
        transform=fig.transFigure,
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--row-dirs", nargs="+", required=True,
                   help="one output dir per row, in top-to-bottom order")
    p.add_argument("--row-labels", nargs="+", required=True,
                   help="left-side label per row; must match --row-dirs length")
    p.add_argument("--iters", default="0,20,40,60,80,100")
    p.add_argument("--output", required=True,
                   help="output PDF (a sibling .png is also saved)")
    p.add_argument("--target-size", type=int, default=0,
                   help="resize every cell to this many pixels per side; "
                        "0 = max size across all inputs")
    p.add_argument("--resample", default="nearest",
                   choices=["auto", "nearest", "bilinear", "bicubic", "lanczos"],
                   help="resampling filter (default nearest)")
    p.add_argument("--cell-size", type=float, default=1.6,
                   help="approximate inches per cell")
    p.add_argument("--hspace", type=float, default=0.5,
                   help="vertical gap between rows, as fraction of axes height")
    p.add_argument("--wspace", type=float, default=0.08,
                   help="horizontal gap between columns, as fraction of axes width")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--save-resized", action="store_true",
                   help="also save each resized cell as a separate PNG under "
                        "<row_dir>/resized_<target>_<resample>/")
    args = p.parse_args()

    if len(args.row_dirs) != len(args.row_labels):
        raise ValueError("--row-dirs and --row-labels must have equal length")

    iters = [int(x) for x in args.iters.split(",") if x.strip()]
    n_rows = len(args.row_dirs)
    n_cols = 1 + len(iters)

    # Load everything, then choose a common pixel size.
    rows_imgs: List[List[Image.Image]] = []
    for row_dir in args.row_dirs:
        imgs = _row_images(row_dir, iters)
        print(f"loaded {row_dir}: original={imgs[0].size}, iters at {imgs[1].size}")
        rows_imgs.append(imgs)

    if args.target_size > 0:
        target = args.target_size
    else:
        target = max(max(im.size) for row in rows_imgs for im in row)
    print(f"target cell pixel size = {target} | resample = {args.resample}")

    rows_imgs = [[_resize_to(im, target, args.resample) for im in row]
                 for row in rows_imgs]

    # Save resized cells separately if requested.
    if args.save_resized:
        for row_dir, imgs in zip(args.row_dirs, rows_imgs):
            out_dir = os.path.join(row_dir, f"resized_{target}_{args.resample}")
            os.makedirs(out_dir, exist_ok=True)
            imgs[0].save(os.path.join(out_dir, "original.png"))
            for i, it in enumerate(iters):
                imgs[i + 1].save(os.path.join(out_dir, f"iter_{it:03d}.png"))
            print(f"saved resized cells -> {out_dir}/")

    # Grow the figure vertically to absorb the extra hspace so each cell
    # keeps roughly the requested cell-size in inches.
    fig_w = args.cell_size * n_cols * (1.0 + args.wspace)
    fig_h = args.cell_size * (n_rows + (n_rows - 1) * args.hspace) + 0.4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    col_titles = ["Original"] + [f"Iter {it}" for it in iters]

    # No cell borders.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for r, imgs in enumerate(rows_imgs):
        for c in range(n_cols):
            # numpy.asarray makes the PIL -> matplotlib path explicit.
            axes[r, c].imshow(np.asarray(imgs[c]))

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=12)
    for r, label in enumerate(args.row_labels):
        axes[r, 0].set_ylabel(label, fontsize=11)

    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.92, bottom=0.05,
        wspace=args.wspace, hspace=args.hspace,
    )

    # Solid divider between the Original column and the Iter columns.
    if n_cols >= 2:
        _add_vertical_divider(fig, axes, between=0)

    # Dashed divider between every pair of rows.
    for r in range(n_rows - 1):
        _add_horizontal_dashed_divider(fig, axes, between=r)

    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved {args.output} ({os.path.getsize(args.output):,} bytes)")

    base, ext = os.path.splitext(args.output)
    if ext.lower() != ".png":
        sibling = base + ".png"
        fig.savefig(sibling, dpi=args.dpi)
        print(f"Saved {sibling} ({os.path.getsize(sibling):,} bytes)")

    plt.close(fig)


if __name__ == "__main__":
    main()
