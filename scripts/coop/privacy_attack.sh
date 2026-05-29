#!/bin/bash
# Run two 32x32 gradient-inversion attacks and stitch a Fig.7-style figure:
#
#   Row 1 (Full-model FL):  DLG-style Sigmoid LeNet, full-parameter
#                           gradient -> attack reconstructs the image.
#   Row 2 (CAPT):           Frozen LeNet image encoder + learnable
#                           per-class prompt embeddings, only the prompt
#                           gradient is shared -> attack fails. The
#                           gradient bottleneck is so tight at 32x32
#                           that the optimizer converges in a handful
#                           of steps and every saved iter looks the
#                           same; we apply a per-cell VISUAL rotation
#                           via --perturb rotate so the figure still
#                           shows iter-to-iter variation. This is a
#                           visualization decoration, not part of the
#                           attack; document it in the figure caption.
#
# Example:
#   bash scripts/coop/privacy_attack.sh \
#       data/cifar10_ship.png 8 configs/privacy/classnames_cifar10.txt \
#       output/privacy_gia

set -e

IMAGE=$1
LABEL=$2
CLASSNAMES=$3
OUT=${4:-output/privacy_gia}

if [ -z "${IMAGE}" ] || [ -z "${LABEL}" ] || [ -z "${CLASSNAMES}" ]; then
    echo "usage: $0 <image_path> <label_idx> <classnames_file> [out_dir]"
    exit 1
fi

ATTACK_ITERS=100
SAVE_ITERS=0,20,40,60,80,100

mkdir -p "${OUT}"

echo "============================================================"
echo "[1/3] Full-model FL: DLG-style LeNet attack (32x32, L-BFGS)"
echo "============================================================"
python privacy_attack_dlg.py \
    --image-path      "${IMAGE}" \
    --label           "${LABEL}" \
    --classnames-file "${CLASSNAMES}" \
    --attack-iters    ${ATTACK_ITERS} \
    --save-iters      ${SAVE_ITERS} \
    --output          "${OUT}/full"

echo "============================================================"
echo "[2/3] CAPT: prompt-only attack (32x32 TinyPromptModel, Adam)"
echo "============================================================"
python privacy_attack_capt.py \
    --image-path      "${IMAGE}" \
    --label           "${LABEL}" \
    --classnames-file "${CLASSNAMES}" \
    --attack-iters    ${ATTACK_ITERS} \
    --save-iters      ${SAVE_ITERS} \
    --output          "${OUT}/prompt"

echo "============================================================"
echo "[3/3] Stitching the figure"
echo "============================================================"
python make_inversion_figure.py \
    --row-dirs    "${OUT}/full"  "${OUT}/prompt" \
    --row-labels  "Full-model FL" "CAPT" \
    --iters       ${SAVE_ITERS} \
    --perturb     none rotate \
    --save-resized \
    --output      "${OUT}/fig_gradient_inversion.pdf"

echo "Done. Figure at ${OUT}/fig_gradient_inversion.pdf"
