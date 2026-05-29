#!/bin/bash
# Run two gradient-inversion attacks and stitch a Fig.7-style figure:
#
#   Row 1 (Full-model FL):  DLG-style Sigmoid LeNet on 32x32 CIFAR-10,
#                           full-parameter gradient -> attack reconstructs.
#                           Saved PNGs are 32x32 native, displayed at
#                           target_size with NEAREST so the pixel-art
#                           look is preserved.
#   Row 2 (CAPT):           CLIP RN50 + CoOp at 224x224, prompt-only
#                           gradient -> attack fails. Saved PNGs are
#                           224x224, downscaled to target_size with
#                           block_max so the per-iter noise variation
#                           survives the resize instead of collapsing
#                           to a uniform-looking subsample.
#
# Why CLIP for the CAPT row instead of a 32x32 TinyPromptModel?
# At 32x32 the prompt gradient (640 floats) is way under-determined
# vs 3072 pixels; the attack converges to a trivial gradient match in
# 1-2 L-BFGS steps and every saved iter looks identical. At 224x224 the
# attack makes visible, per-iter progress in noise space.
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
echo "[2/3] CAPT: CoOp prompt-only gradient attack (CLIP RN50, 224)"
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
    --row-dirs    "${OUT}/full"  "${OUT}/prompt" \
    --row-labels  "Full-model FL" "CAPT" \
    --iters       ${SAVE_ITERS} \
    --target-size 32 \
    --resample    nearest block_max \
    --save-resized \
    --output      "${OUT}/fig_gradient_inversion.pdf"

echo "Done. Figure at ${OUT}/fig_gradient_inversion.pdf"
