#!/bin/bash
# Run two gradient-inversion attacks and stitch a Fig.7-style figure:
#
#   Row 1 (Full-model FL):  small ResNet-18 on 32x32 CIFAR-10, full
#                           parameter gradient -> attack reconstructs.
#   Row 2 (PromptFL):       CLIP + CoOp, prompt-only gradient ->
#                           attack fails to reconstruct.
#
# Two different models on purpose: that is exactly the contrast PromptFL
# Fig. 7 draws (model-based FL vs prompt-only FL). Both rows use the
# same source image so the leftmost "Original" column is consistent.
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
echo "[2/3] PromptFL: CoOp prompt-only gradient attack (CLIP RN50)"
echo "============================================================"
python privacy_attack_coop.py \
    --image-path      "${IMAGE}" \
    --label           "${LABEL}" \
    --classnames-file "${CLASSNAMES}" \
    --signal          prompt \
    --backbone        RN50 \
    --attack-iters    ${ATTACK_ITERS} \
    --save-iters      ${SAVE_ITERS} \
    --output          "${OUT}/prompt"

echo "============================================================"
echo "[3/3] Stitching the figure"
echo "============================================================"
python make_inversion_figure.py \
    --row-dirs   "${OUT}/full"  "${OUT}/prompt" \
    --row-labels "Full-model FL" "PromptFL" \
    --iters      ${SAVE_ITERS} \
    --output     "${OUT}/fig_gradient_inversion.pdf"

echo "Done. Figure at ${OUT}/fig_gradient_inversion.pdf"
