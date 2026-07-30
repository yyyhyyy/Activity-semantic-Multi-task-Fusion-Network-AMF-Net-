#!/usr/bin/env bash
set -euo pipefail

cd /home/ma-user/work/test

FOLDS="${FOLDS:-5}"
EPOCHS="${EPOCHS:-30}"
SOCRATIC_RECALL_BOOST="${SOCRATIC_RECALL_BOOST:-0.08}"
EXTRA_ARGS="$*"
ROOT="/home/ma-user/work/test/outputs/paper_experiments"

COMMON_ARGS=(
  --data_root "/home/ma-user/work/test/data"
  --backbone swin3d_t
  --fusion mlp
  --sampling uniform
  --folds "${FOLDS}"
  --epochs "${EPOCHS}"
  --batch_size 8
  --clip_len 16
  --stride 16
  --image_size 112
  --num_workers 8
  --eval_num_workers 4
  --device npu
  --amp
  --use_wcls
  --discuss_loss_weight 5.0
  --backbone_adapter evidence_st_conv
  --feature_adapter ms_lka_rare
  --adapter_reduction 4
  --adapter_scale 0.10
  --adapter_dropout 0.05
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
  --data_temporal_rescue_eval
  --data_temporal_window 64
  --data_temporal_score_threshold 0.22
  --data_temporal_neighbor_threshold 0.30
  --data_temporal_max_margin 4.5
  --question_temporal_rescue_eval
  --question_temporal_window 64
  --question_temporal_score_threshold 0.38
  --question_temporal_neighbor_threshold 0.30
  --question_temporal_max_margin 4.5
  --pedagogical_template_adapter
  --pedagogical_template_scale 0.45
  --scene_desk_constraint_adapter
  --scene_desk_constraint_scale 0.45
  --pedagogical_prior_adapter
  --pedagogical_prior_scale 0.18
  --pedagogical_consistency_weight 0.35
  --debate_temper_scale 0.0
  --socratic_cap_scale 0.0
  --socratic_recall_boost_scale "${SOCRATIC_RECALL_BOOST}"
  --debate_temporal_rescue_eval
  --debate_temporal_window 64
  --debate_temporal_score_threshold 0.24
  --debate_temporal_neighbor_threshold 0.20
  --debate_temporal_max_margin 6.0
  --guide_question_relaxed_eval
  --guide_location_rule_videos 7 9
  --selection_metric paper_balanced
)

python train_ablation.py \
  "${COMMON_ARGS[@]}" \
  --split_mode purged_temporal \
  --purge_neighbors 2 \
  --out_dir "${ROOT}/01_robust_purged_temporal" \
  ${EXTRA_ARGS}

python train_ablation.py \
  "${COMMON_ARGS[@]}" \
  --split_mode video_holdout \
  --out_dir "${ROOT}/02_robust_video_holdout_diag" \
  ${EXTRA_ARGS}

python train_ablation.py \
  "${COMMON_ARGS[@]}" \
  --split_mode temporal \
  --discuss_eval_mode video_mean \
  --out_dir "${ROOT}/03_robust_video_mean_diag" \
  ${EXTRA_ARGS}
