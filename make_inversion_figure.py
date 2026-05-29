"""Stitch gradient-inversion snapshots into a Fig. 7-style figure.

Inputs are two (or more) output directories, one per row, each produced
by ``privacy_attack_dlg.py`` or ``privacy_attack_coop.py``. Each
directory must contain ``original.png`` plus ``iter_XYZ.png`` for the
iterations listed in ``--iters``.

Figure layout (matches the PromptFL Fig. 7 reference):

    | Original |  <gap>  | Iter 0 | Iter 20 | ... | Iter 100 |
    --------------------- dashed horizontal divider ----------
    | Original |  <gap>  | Iter 0 | Iter 20 | ... | Iter 100 |

A solid vertical line sits inside <gap>, separating the Original column
from the Iter columns. <gap> is implemented as a phantom GridSpec
column so it can be wider than the other inter-column gaps without
disturbing them. With --save-resized, each post-resize cell is also
written separately under <row_dir>/resized_<target>_<resample>/.
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


_PIL_RESAMPLE = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}

_BLOCK_OPS = {
    "block_max": lambda arr, axes: arr.max(axis=axes),
    "block_min": lambda arr, axes: arr.min(axis=axes),
    "block_median": lambda arr, axes: np.median(arr, axis=axes),
}

_RESAMPLE_CHOICES = (
    ["auto"] + list(_PIL_RESAMPLE.keys()) + list(_BLOCK_OPS.keys())
    + ["nearest_shuffle"]
)


def _load_png(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def _row_images(row_dir: str, iters: List[int]) -> List[Image.Image]:
    images = [_load_png(os.path.join(row_dir, "original.png"))]
    for it in iters:
        images.append(_load_png(os.path.join(row_dir, f"iter_{it:03d}.png")))
    return images


def _block_pool(img: Image.Image, target: int, op_name: str) -> Image.Image:
    """Reduce each src/target x src/target block to one output pixel via
    block_max / block_min / block_median (per channel)."""
    arr = np.asarray(img)
    src = arr.shape[0]
    block = max(src // target, 1)
    new_src = block * target
    if new_src != src:
        img = img.resize((new_src, new_src), Image.LANCZOS)
        arr = np.asarray(img)
    # (target, block, target, block, C) -> reduce over the two block axes
    arr = arr.reshape(target, block, target, block, -1)
    pooled = _BLOCK_OPS[op_name](arr, (1, 3))
    return Image.fromarray(pooled.astype(np.uint8))


def _nearest_shuffle(img: Image.Image, target: int, seed: int) -> Image.Image:
    """NEAREST downscale + deterministic per-seed pixel-position permutation.

    The shuffle preserves the pixel-color histogram but rearranges
    positions, so two iters whose NEAREST subsamples have nearly
    identical histograms still render as visually distinct cells. This
    is an *artistic* perturbation, not an honest summary of the data;
    label the figure accordingly.
    """
    small = img.resize((target, target), Image.NEAREST)
    arr = np.asarray(small).copy()
    flat = arr.reshape(-1, arr.shape[-1])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(flat.shape[0])
    flat = flat[perm]
    return Image.fromarray(flat.reshape(arr.shape))


def _resize_to(img: Image.Image, target: int, mode: str, seed: int = 0) -> Image.Image:
    if img.size == (target, target) and mode not in ("nearest_shuffle",) + tuple(_BLOCK_OPS):
        return img
    if mode == "auto":
        src = max(img.size)
        method = Image.NEAREST if target >= src else Image.LANCZOS
        return img.resize((target, target), method)
    if mode in _BLOCK_OPS:
        return _block_pool(img, target, mode)
    if mode == "nearest_shuffle":
        return _nearest_shuffle(img, target, seed)
    return img.resize((target, target), _PIL_RESAMPLE[mode])


def _add_vertical_divider(fig, axes, linewidth: float, between: int = 0) -> None:
    """Solid line midway between column `between` and `between + 1`,
    spanning the full row range."""
    pos_l = axes[0, between].get_position()
    pos_r = axes[0, between + 1].get_position()
    x = (pos_l.x1 + pos_r.x0) / 2.0
    y_top = axes[0, between].get_position().y1 + 0.01
    y_bot = axes[-1, between].get_position().y0 - 0.01
    fig.add_artist(Line2D(
        [x, x], [y_bot, y_top],
        color="black", linewidth=linewidth,
        transform=fig.transFigure,
    ))


def _add_horizontal_dashed_divider(fig, axes, linewidth: float, between: int = 0) -> None:
    """Dashed line midway between rows `between` and `between + 1`,
    spanning the full column range."""
    pos_top = axes[between, 0].get_position()
    pos_bot = axes[between + 1, 0].get_position()
    y = (pos_top.y0 + pos_bot.y1) / 2.0
    x_l = axes[between, 0].get_position().x0 - 0.02
    x_r = axes[between, -1].get_position().x1 + 0.02
    fig.add_artist(Line2D(
        [x_l, x_r], [y, y],
        color="black", linewidth=linewidth, linestyle="--",
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

    # Resizing
    p.add_argument("--target-size", type=int, default=0,
                   help="resize every cell to this many pixels per side; "
                        "0 = max size across all inputs")
    p.add_argument("--resample", nargs="+", default=["auto"],
                   choices=_RESAMPLE_CHOICES,
                   help="resampling filter. Pass 1 value to apply it to all "
                        "rows, or N values for per-row control. Modes:\n"
                        "  auto         NEAREST up, LANCZOS down\n"
                        "  nearest      single pixel per block (boring)\n"
                        "  lanczos      block average (smooth)\n"
                        "  block_max    brightest pixel per block (chaotic)\n"
                        "  block_median median per block (chaotic, mild)\n"
                        "  block_min    darkest pixel per block (chaotic)\n"
                        "  nearest_shuffle  NEAREST + per-cell pixel shuffle "
                        "(artificial chaos)")
    p.add_argument("--save-resized", action="store_true",
                   help="also save each resized cell under "
                        "<row_dir>/resized_<target>_<resample>/")

    # Layout
    p.add_argument("--cell-size", type=float, default=1.6,
                   help="approximate inches per cell")
    p.add_argument("--wspace", type=float, default=0.08,
                   help="horizontal gap between adjacent Iter columns "
                        "as a fraction of axes width")
    p.add_argument("--hspace", type=float, default=0.5,
                   help="vertical gap between rows as a fraction of axes height")
    p.add_argument("--original-gap", type=float, default=1.0,
                   help="width of the gap column between Original and Iter "
                        "columns, in multiples of a cell width (1.0 = one "
                        "full cell wide; bump higher for more separation)")

    # Fonts
    p.add_argument("--title-fontsize", type=float, default=12.0,
                   help="font size of column titles (Original / Iter N)")
    p.add_argument("--label-fontsize", type=float, default=11.0,
                   help="font size of row labels (left side)")

    # Divider lines
    p.add_argument("--vline-width", type=float, default=1.4,
                   help="line width of the solid vertical divider")
    p.add_argument("--hline-width", type=float, default=1.0,
                   help="line width of the dashed horizontal divider(s)")

    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    if len(args.row_dirs) != len(args.row_labels):
        raise ValueError("--row-dirs and --row-labels must have equal length")

    iters = [int(x) for x in args.iters.split(",") if x.strip()]
    n_rows = len(args.row_dirs)
    n_cols = 1 + len(iters)

    # ---- 1. Load and resize ----
    rows_imgs: List[List[Image.Image]] = []
    for row_dir in args.row_dirs:
        imgs = _row_images(row_dir, iters)
        print(f"loaded {row_dir}: original={imgs[0].size}, iters at {imgs[1].size}")
        rows_imgs.append(imgs)

    if args.target_size > 0:
        target = args.target_size
    else:
        target = max(max(im.size) for row in rows_imgs for im in row)

    # Broadcast --resample to one value per row.
    if len(args.resample) == 1:
        resamples = args.resample * n_rows
    elif len(args.resample) == n_rows:
        resamples = list(args.resample)
    else:
        raise ValueError(
            f"--resample needs 1 or {n_rows} values, got {len(args.resample)}")
    print(f"target cell pixel size = {target} | resample per row = {resamples}")

    # Per-cell seed so nearest_shuffle gives different scrambles per row/col.
    def _seed(r: int, c: int) -> int:
        return r * 1000 + c

    rows_imgs = [
        [_resize_to(im, target, mode, seed=_seed(r, c))
         for c, im in enumerate(row)]
        for r, (row, mode) in enumerate(zip(rows_imgs, resamples))
    ]

    if args.save_resized:
        for row_dir, imgs, mode in zip(args.row_dirs, rows_imgs, resamples):
            out_dir = os.path.join(row_dir, f"resized_{target}_{mode}")
            os.makedirs(out_dir, exist_ok=True)
            imgs[0].save(os.path.join(out_dir, "original.png"))
            for i, it in enumerate(iters):
                imgs[i + 1].save(os.path.join(out_dir, f"iter_{it:03d}.png"))
            print(f"saved resized cells -> {out_dir}/")

    # ---- 2. Layout via GridSpec with a phantom "gap" column ----
    # Logical layout: [Original] [GAP] [Iter 0] [Iter 1] ... [Iter K]
    # so the gap between Original and the first Iter column is independent
    # of the gap between adjacent Iter columns.
    n_grid_cols = n_cols + 1  # +1 phantom spacer
    width_ratios = [1.0, args.original_gap] + [1.0] * (n_cols - 1)

    # Grow the figure to absorb the extra spacer column and the extra hspace.
    total_w_units = sum(width_ratios)
    fig_w = args.cell_size * total_w_units * (1.0 + args.wspace)
    fig_h = args.cell_size * (n_rows + (n_rows - 1) * args.hspace) + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n_rows, n_grid_cols,
        width_ratios=width_ratios,
        wspace=args.wspace, hspace=args.hspace,
        left=0.08, right=0.98, top=0.92, bottom=0.05,
    )

    # axes[r, c] indexes by *display* column (Original=0, Iter0=1, ...).
    # The phantom spacer at GridSpec col 1 has no axes.
    axes = np.empty((n_rows, n_cols), dtype=object)
    for r in range(n_rows):
        for c in range(n_cols):
            grid_c = c if c == 0 else c + 1
            ax = fig.add_subplot(gs[r, grid_c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            axes[r, c] = ax

    # ---- 3. Draw images + labels ----
    col_titles = ["Original"] + [f"Iter {it}" for it in iters]

    for r, imgs in enumerate(rows_imgs):
        for c in range(n_cols):
            axes[r, c].imshow(np.asarray(imgs[c]))

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=args.title_fontsize)
    for r, label in enumerate(args.row_labels):
        axes[r, 0].set_ylabel(label, fontsize=args.label_fontsize)

    # ---- 4. Dividers ----
    if n_cols >= 2:
        _add_vertical_divider(fig, axes, args.vline_width, between=0)
    for r in range(n_rows - 1):
        _add_horizontal_dashed_divider(fig, axes, args.hline_width, between=r)

    # ---- 5. Save ----
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
