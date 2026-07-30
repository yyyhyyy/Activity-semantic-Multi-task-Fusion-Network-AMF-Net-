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
BASELINE_BACKBONES="${BASELINE_BACKBONES:-s3d r3d_18 mc3_18 r2plus1d_18 swin3d_t}"
FULL_BACKBONE="${FULL_BACKBONE:-swin3d_t}"
FULL_MODEL_NAME="${FULL_MODEL_NAME:-${FULL_BACKBONE}_full}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_FULL="${RUN_FULL:-1}"
ROOT="${ROOT:-/home/ma-user/work/test/outputs/paper_experiments/plain_baseline_vs_full}"
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

run_plain_baseline () {
  local backbone="$1"
  local out_name="$2"
  python train_ablation.py \
    "${COMMON_ARGS[@]}" \
    --discuss_only \
    --fusion none \
    --backbone "${backbone}" \
    --selection_metric balanced \
    --video_bag_loss_weight 0.0 \
    --pair_balance_loss_weight 0.0 \
    --guide_specific_loss_weight 0.0 \
    --data_specific_loss_weight 0.0 \
    --behavior_evidence_loss_weight 0.0 \
    --scene_desk_constraint_loss_weight 0.0 \
    --data_behavior_fewshot_loss_weight 0.0 \
    --pair_margin_loss_weight 0.0 \
    --out_dir "${ROOT}/baselines/${out_name}" \
    ${EXTRA_ARGS}
}

run_full_model () {
  python train_ablation.py \
    "${COMMON_ARGS[@]}" \
    --fusion mlp \
    --backbone "${FULL_BACKBONE}" \
    --selection_metric paper_balanced \
    --out_dir "${ROOT}/full_model/${FULL_MODEL_NAME}" \
    --data_specific_head \
    --data_specific_scale 0.34 \
    --data_specific_loss_weight 2.2 \
    --data_specific_guard_margin 0.80 \
    --data_behavior_fewshot_loss_weight 1.7 \
    --data_debate_conflict_loss_weight 1.5 \
    --data_debate_conflict_margin 0.80 \
    --data_router_scale 0.50 \
    --data_router_threshold 0.22 \
    --data_router_suppress_scale 0.62 \
    --data_router_margin 0.45 \
    --data_evidence_boost_scale 0.14 \
    --pedagogical_template_adapter \
    --pedagogical_template_scale 0.45 \
    --scene_desk_constraint_adapter \
    --scene_desk_constraint_scale 0.45 \
    --pedagogical_prior_adapter \
    --pedagogical_prior_scale 0.18 \
    --pedagogical_consistency_weight 0.35 \
    --socratic_recall_boost_scale 0.08 \
    --guide_location_rule_videos 7 9 \
    ${EXTRA_ARGS}
}

echo "[plain_baseline_vs_full] output root: ${ROOT}"
echo "[plain_baseline_vs_full] split mode: ${SPLIT_MODE}, purge neighbors: ${PURGE_NEIGHBORS}"
echo "[plain_baseline_vs_full] full model backbone: ${FULL_BACKBONE} -> ${FULL_MODEL_NAME}"

if [[ "${RUN_BASELINES}" != "0" ]]; then
  echo "[plain_baseline_vs_full] running plain discuss-only backbone baselines: ${BASELINE_BACKBONES}"
  for backbone in ${BASELINE_BACKBONES}; do
    case "${backbone}" in
      s3d) run_plain_baseline s3d "B1_s3d_plain" ;;
      r3d_18) run_plain_baseline r3d_18 "B2_r3d_18_plain" ;;
      mc3_18) run_plain_baseline mc3_18 "B3_mc3_18_plain" ;;
      r2plus1d_18) run_plain_baseline r2plus1d_18 "B4_r2plus1d_18_plain" ;;
      swin3d_t) run_plain_baseline swin3d_t "B5_swin3d_t_plain" ;;
      *) echo "[plain_baseline_vs_full] unknown baseline backbone: ${backbone}" >&2; exit 2 ;;
    esac
  done
else
  echo "[plain_baseline_vs_full] skipping plain baselines; existing baseline outputs under ROOT will be reused by the report."
fi

if [[ "${RUN_FULL}" != "0" ]]; then
  echo "[plain_baseline_vs_full] running full model with proposed components"
  run_full_model
else
  echo "[plain_baseline_vs_full] skipping full model run."
fi

python paper_experiments/baseline_vs_full_report.py --root "${ROOT}"
