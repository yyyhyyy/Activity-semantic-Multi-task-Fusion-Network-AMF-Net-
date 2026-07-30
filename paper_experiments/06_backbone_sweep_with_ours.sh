#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/test

FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CLIP_LEN="${CLIP_LEN:-16}"
STRIDE="${STRIDE:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-112}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
CANDIDATE_BACKBONES="${CANDIDATE_BACKBONES:-s3d r3d_18 mc3_18 r2plus1d_18 swin3d_t}"
ROOT="${ROOT:-/home/ma-user/work/test/outputs/paper_experiments/backbone_sweep_with_ours}"
SPLIT_MODE="${SPLIT_MODE:-temporal}"
PURGE_NEIGHBORS="${PURGE_NEIGHBORS:-2}"
EXTRA_ARGS="$*"

COMMON_ARGS=(
  --data_root "/home/ma-user/work/test/data"
  --sampling uniform
  --split_mode "${SPLIT_MODE}"
  --purge_neighbors "${PURGE_NEIGHBORS}"
  --folds "${FOLDS}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --clip_len "${CLIP_LEN}"
  --stride "${STRIDE}"
  --image_size "${IMAGE_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --eval_num_workers "${EVAL_NUM_WORKERS}"
  --device npu
  --amp
  --use_wcls
  --discuss_loss_weight 5.0
  --report_strict_primary
  --print_raw_discuss_table
  --save_discuss_predictions
)

run_ours () {
  local backbone="$1"
  local out_name="$2"
  python train_ablation.py     "${COMMON_ARGS[@]}"     --fusion mlp     --backbone "${backbone}"     --selection_metric paper_balanced     --out_dir "${ROOT}/runs/${out_name}"     --data_specific_head     --data_specific_scale 0.34     --data_specific_loss_weight 2.2     --data_specific_guard_margin 0.80     --data_behavior_fewshot_loss_weight 1.7     --data_debate_conflict_loss_weight 1.5     --data_debate_conflict_margin 0.80     --data_router_scale 0.50     --data_router_threshold 0.22     --data_router_suppress_scale 0.62     --data_router_margin 0.45     --data_evidence_boost_scale 0.14     --pedagogical_template_adapter     --pedagogical_template_scale 0.45     --scene_desk_constraint_adapter     --scene_desk_constraint_scale 0.45     --pedagogical_prior_adapter     --pedagogical_prior_scale 0.18     --pedagogical_consistency_weight 0.35     --socratic_recall_boost_scale 0.08     --guide_location_rule_videos 7 9     ${EXTRA_ARGS}
}

echo "[backbone_sweep_with_ours] output root: ${ROOT}"
echo "[backbone_sweep_with_ours] split mode: ${SPLIT_MODE}, purge neighbors: ${PURGE_NEIGHBORS}"
echo "[backbone_sweep_with_ours] candidate backbones: ${CANDIDATE_BACKBONES}"

mkdir -p "${ROOT}/runs"

for backbone in ${CANDIDATE_BACKBONES}; do
  case "${backbone}" in
    s3d) run_ours s3d "ours_s3d" ;;
    r3d_18) run_ours r3d_18 "ours_r3d_18" ;;
    mc3_18) run_ours mc3_18 "ours_mc3_18" ;;
    r2plus1d_18) run_ours r2plus1d_18 "ours_r2plus1d_18" ;;
    swin3d_t) run_ours swin3d_t "ours_swin3d_t" ;;
    *) echo "[backbone_sweep_with_ours] unknown backbone: ${backbone}" >&2; exit 2 ;;
  esac
done

python paper_experiments/backbone_sweep_report.py --root "${ROOT}"
