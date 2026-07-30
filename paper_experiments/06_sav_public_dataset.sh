#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/test

SAV_ROOT="${SAV_ROOT:-/home/ma-user/work/test/data/SAV}"
ROOT="${ROOT:-/home/ma-user/work/test/outputs/paper_experiments/sav_public_dataset}"
BACKBONES="${BACKBONES:-s3d r3d_18 mc3_18 r2plus1d_18 swin3d_t}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CLIP_LEN="${CLIP_LEN:-16}"
SAMPLE_RATE="${SAMPLE_RATE:-1}"
IMAGE_SIZE="${IMAGE_SIZE:-112}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
VAL_SPLIT="${VAL_SPLIT:-val}"
FRAMES_DIR="${FRAMES_DIR:-frames}"
VIDEOS_DIR="${VIDEOS_DIR:-clips}"
ANNOTATIONS_DIR="${ANNOTATIONS_DIR:-annotations}"
FRAME_LIST_DIR="${FRAME_LIST_DIR:-frame_list}"
LABEL_MAP="${LABEL_MAP:-}"
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-0}"
EXTRA_ARGS="$*"

COMMON_ARGS=(
  --data_root "${SAV_ROOT}"
  --train_split "${TRAIN_SPLIT}"
  --val_split "${VAL_SPLIT}"
  --frames_dir "${FRAMES_DIR}"
  --videos_dir "${VIDEOS_DIR}"
  --annotations_dir "${ANNOTATIONS_DIR}"
  --frame_list_dir "${FRAME_LIST_DIR}"
  --clip_len "${CLIP_LEN}"
  --sample_rate "${SAMPLE_RATE}"
  --image_size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --eval_num_workers "${EVAL_NUM_WORKERS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --max_samples_per_split "${MAX_SAMPLES_PER_SPLIT}"
  --device npu
  --amp
)

if [[ -n "${LABEL_MAP}" ]]; then
  COMMON_ARGS+=(--label_map "${LABEL_MAP}")
fi

echo "[SAV] data root: ${SAV_ROOT}"
echo "[SAV] output root: ${ROOT}"
echo "[SAV] backbones: ${BACKBONES}"

for backbone in ${BACKBONES}; do
  out_name="${backbone}_sav"
  echo "[SAV] running ${backbone} -> ${ROOT}/${out_name}"
  python train_sav.py \
    "${COMMON_ARGS[@]}" \
    --backbone "${backbone}" \
    --out_dir "${ROOT}/${out_name}" \
    ${EXTRA_ARGS}
done

python paper_experiments/sav_report.py --root "${ROOT}"

