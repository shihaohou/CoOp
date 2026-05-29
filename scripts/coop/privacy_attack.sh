#!/bin/bash
# Run the gradient-inversion attack on (1) the full CLIP visual encoder
# gradient and (2) the prompt-only gradient, then stitch a Fig.7-style
# figure.
#
# Example:
#   bash scripts/coop/privacy_attack.sh \
#       /path/to/sample.jpg 3 configs/privacy/classnames_cifar10.txt \
#       output/privacy_gia
#
# Notes:
#   - $1 must be a real image path on the H800 box.
#   - $2 is the ground-truth class index in $3 (0-indexed).
#   - Output goes under $4 (default: output/privacy_gia).

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
BACKBONE=RN50

mkdir -p "${OUT}"

echo "============================================================"
echo "[1/3] Full-model FL (CLIP visual encoder) gradient attack"
echo "============================================================"
python privacy_attack_coop.py \
    --image-path  "${IMAGE}" \
    --label        "${LABEL}" \
    --classnames-file "${CLASSNAMES}" \
    --signal       full \
    --backbone     "${BACKBONE}" \
    --attack-iters ${ATTACK_ITERS} \
    --save-iters   ${SAVE_ITERS} \
    --output       "${OUT}/full"

echo "============================================================"
echo "[2/3] PromptFL / CoOp prompt-only gradient attack"
echo "============================================================"
python privacy_attack_coop.py \
    --image-path  "${IMAGE}" \
    --label        "${LABEL}" \
    --classnames-file "${CLASSNAMES}" \
    --signal       prompt \
    --backbone     "${BACKBONE}" \
    --attack-iters ${ATTACK_ITERS} \
    --save-iters   ${SAVE_ITERS} \
    --output       "${OUT}/prompt"

echo "============================================================"
echo "[3/3] Stitching the figure"
echo "============================================================"
python make_inversion_figure.py \
    --row-dirs   "${OUT}/full"  "${OUT}/prompt" \
    --row-labels "Full-model FL" "PromptFL" \
    --iters      ${SAVE_ITERS} \
    --output     "${OUT}/fig_gradient_inversion.pdf"

echo "Done. Figure at ${OUT}/fig_gradient_inversion.pdf"
