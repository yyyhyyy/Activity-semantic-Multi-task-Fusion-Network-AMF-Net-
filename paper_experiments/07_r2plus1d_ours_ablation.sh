#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/test

BACKBONE="${BACKBONE:-r2plus1d_18}"
FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CLIP_LEN="${CLIP_LEN:-16}"
STRIDE="${STRIDE:-16}"
IMAGE_SIZE="${IMAGE_SIZE:-112}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-4}"
ROOT="${ROOT:-/home/ma-user/work/test/outputs/paper_experiments/r2plus1d_ours_ablation}"
SPLIT_MODE="${SPLIT_MODE:-temporal}"
PURGE_NEIGHBORS="${PURGE_NEIGHBORS:-2}"
ABLATION_MODE="${ABLATION_MODE:-paper_core}"
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

DATA_HEAD_ARGS=(
  --data_specific_head
  --data_specific_scale 0.34
  --data_specific_loss_weight 2.2
  --data_specific_guard_margin 0.80
  --data_behavior_fewshot_loss_weight 1.7
  --data_debate_conflict_loss_weight 1.5
  --data_debate_conflict_margin 0.80
  --data_router_scale 0.50
  --data_router_threshold 0.22
  --data_router_suppress_scale 0.62
  --data_router_margin 0.45
  --data_evidence_boost_scale 0.14
)

PED_ARGS=(
  --pedagogical_template_adapter
  --pedagogical_template_scale 0.45
  --scene_desk_constraint_adapter
  --scene_desk_constraint_scale 0.45
  --pedagogical_prior_adapter
  --pedagogical_prior_scale 0.18
  --pedagogical_consistency_weight 0.35
)

TEMPORAL_ARGS=(
  --socratic_recall_boost_scale 0.08
  --guide_location_rule_videos 7 9
)

run_variant () {
  local name="$1"
  shift
  echo "[r2plus1d_ours_ablation] running ${name}"
  python train_ablation.py \
    "${COMMON_ARGS[@]}" \
    --backbone "${BACKBONE}" \
    --out_dir "${ROOT}/runs/${name}" \
    "$@" \
    ${EXTRA_ARGS}
}

echo "[r2plus1d_ours_ablation] output root: ${ROOT}"
echo "[r2plus1d_ours_ablation] backbone: ${BACKBONE}"
echo "[r2plus1d_ours_ablation] split mode: ${SPLIT_MODE}, purge neighbors: ${PURGE_NEIGHBORS}"
echo "[r2plus1d_ours_ablation] ablation mode: ${ABLATION_MODE}"

mkdir -p "${ROOT}/runs"

run_variant A0_plain \
  --discuss_only \
  --fusion none \
  --selection_metric balanced \
  --video_bag_loss_weight 0.0 \
  --pair_balance_loss_weight 0.0 \
  --guide_specific_loss_weight 0.0 \
  --data_specific_loss_weight 0.0 \
  --behavior_evidence_loss_weight 0.0 \
  --scene_desk_constraint_loss_weight 0.0 \
  --data_behavior_fewshot_loss_weight 0.0 \
  --pair_margin_loss_weight 0.0

run_variant A1_multitask_fusion \
  --fusion mlp \
  --selection_metric paper_balanced

case "${ABLATION_MODE}" in
  paper_core)
    # Paper-facing incremental sequence: only modules retained in PMF-Net are
    # presented as the main path. Optional or unstable modules stay out of the
    # main ablation table unless diagnostic_full is explicitly used.
    run_variant A2_pedagogical_constraints \
      --fusion mlp \
      --selection_metric paper_balanced \
      "${PED_ARGS[@]}"

    run_variant A3_pmfn_net \
      --fusion mlp \
      --selection_metric paper_balanced \
      "${PED_ARGS[@]}" \
      "${TEMPORAL_ARGS[@]}"
    ;;

  diagnostic_full)
    # Full diagnostic sweep: includes optional modules that may be useful for
    # error analysis, but should not be claimed as PMF-Net components if they
    # do not improve the final-backbone result.
    run_variant D2_data_head \
      --fusion mlp \
      --selection_metric paper_balanced \
      "${DATA_HEAD_ARGS[@]}"

    run_variant D3_data_head_pedagogical \
      --fusion mlp \
      --selection_metric paper_balanced \
      "${DATA_HEAD_ARGS[@]}" \
      "${PED_ARGS[@]}"

    run_variant D4_full_with_optional_data_head \
      --fusion mlp \
      --selection_metric paper_balanced \
      "${DATA_HEAD_ARGS[@]}" \
      "${PED_ARGS[@]}" \
      "${TEMPORAL_ARGS[@]}"
    ;;

  *)
    echo "[r2plus1d_ours_ablation] unknown ABLATION_MODE=${ABLATION_MODE}; use paper_core or diagnostic_full" >&2
    exit 2
    ;;
esac

python paper_experiments/ablation_report.py --root "${ROOT}" --mode "${ABLATION_MODE}"