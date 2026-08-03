# AMF-Net: Cognitive-Constructive Group Activity Recognition

This repository contains the code for **AMF-Net (Activity-semantic Multi-task Fusion Network)**, a video understanding framework for recognizing classroom discussion-based teaching activities. The task is formulated as cognitive-constructive group activity recognition: the model predicts the main discussion activity type while using auxiliary classroom semantic labels as pedagogical evidence.

## Repository Contents

The files required for code release and reproducibility are:

- `config.py`: label definitions, task metadata, and label aliases.
- `cvat_parser.py`: CVAT annotation parser.
- `multi_video_dataset.py`: multi-video clip dataset for classroom videos.
- `video_dataset.py`, `dataset.py`: dataset utilities kept for compatibility.
- `experimental_video_models.py`: PMF-Net and backbone construction.
- `semantic_fusion_modules.py`: semantic fusion components.
- `backbone_adapters.py`, `video_models.py`, `classroom_swin3d.py`, `models.py`: video backbone and model utilities.
- `device_utils.py`: CPU/CUDA/NPU device helper functions.
- `train_ablation.py`: main training and evaluation script used for paper experiments.
- `paper_experiments/`: shell scripts for paper-level experiments.
- `requirements.txt`: Python dependencies.

Files related only to manuscript editing, local rendering, or draft generation are not required to reproduce the experiments.

## Dataset

The dataset will be released at:

```text
PMF-Net/data/
```

After downloading, organize the data as follows:

```text
data_root/
  video_001/
    annotations.xml
    images/
      frame_000001.jpg
      frame_000002.jpg
      ...
  video_002/
    annotations.xml
    images/
      ...
  ...
```

Each video folder should contain one CVAT annotation file and the corresponding extracted frames. The code expects frame-level annotations exported in the CVAT video annotation format.

## Label Tasks

The main task is `discuss_type`, with five discussion-based teaching activity classes:

- `question_discuss`: question-driven discussion-based teaching activity
- `guide_discuss`: guided discussion-based teaching activity
- `debate_discuss`: debate-based discussion-based teaching activity
- `socratic_discuss`: Socratic discussion-based teaching activity
- `data_discuss`: data-driven discussion-based teaching activity

Auxiliary tasks include:

- `scene_desk`: classroom desk layout
- `scene_method`: teaching mode
- `scene_inte`: teacher interaction object
- `teacher_act`: teacher action
- `location`: teacher location
- `stu_act`: student action
- `view`: student gaze/view direction

## Environment Setup

Create a Python environment and install dependencies:

```bash
cd /path/to/PMF-Net
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Ascend NPU experiments, use the PyTorch-Ascend environment provided by your server platform. The paper experiments were run with:

```text
pytorch_ascend:pytorch_2.7.1-cann_8.3.rc1-py_3.11-hce_2.0.2509-aarch64-snt9b-20260402124057-b4bd7bb
1 * ascend-snt9b1 | 24 vCPUs | 192 GiB
```

## Quick Data Check

Before training, verify that annotations and frames can be parsed:

```bash
python inspect_data.py --data_root /path/to/data_root
```

If parsing fails, check that each video directory contains `annotations.xml` and an `images/` or frame directory with filenames matching the annotation file.

## Main Paper Experiment

Run PMF-Net with the final R(2+1)D-18 backbone:

```bash
python train_ablation.py \
  --data_root /path/to/data_root \
  --device npu \
  --amp \
  --backbone r2plus1d_18 \
  --folds 5 \
  --epochs 30 \
  --batch_size 4 \
  --clip_len 16 \
  --stride 8 \
  --image_size 112
```

Use `--device cuda` for NVIDIA GPUs and `--device cpu` for CPU-only debugging.

## Paper Experiment Scripts

The `paper_experiments/` directory contains reproducible scripts for the main experimental protocol.

### Backbone Sweep

This experiment compares PMF-Net with different video backbones:

```bash
ROOT=/path/to/outputs/paper_experiments/backbone_sweep \
DATA_ROOT=/path/to/data_root \
bash paper_experiments/06_backbone_sweep_with_ours.sh
```

### Plain Backbone vs. PMF-Net

This experiment compares plain video backbones with the full PMF-Net framework:

```bash
ROOT=/path/to/outputs/paper_experiments/plain_baseline_vs_full \
DATA_ROOT=/path/to/data_root \
SPLIT_MODE=temporal \
bash paper_experiments/05_plain_baseline_vs_full.sh
```

### Purged Temporal Robustness Evaluation

This experiment removes training clips adjacent to validation clips from the same video to reduce temporal leakage:

```bash
ROOT=/path/to/outputs/paper_experiments/robustness \
DATA_ROOT=/path/to/data_root \
SPLIT_MODE=purged_temporal \
bash paper_experiments/05_plain_baseline_vs_full.sh
```

### Controlled Ablation Study

Use the final selected backbone and run the controlled module ablation:

```bash
ROOT=/path/to/outputs/paper_experiments/final_ablation \
DATA_ROOT=/path/to/data_root \
FULL_BACKBONE=r2plus1d_18 \
bash paper_experiments/07_final_ablation.sh
```

If a script name differs in your release package, keep the command structure but replace the script filename with the corresponding ablation script.

## Outputs

Experiment outputs are written under the directory specified by `ROOT` or under `outputs/` by default. Typical outputs include:

- per-fold metrics
- mean and standard deviation of Accuracy and Macro-F1
- class-wise precision, recall, and F1
- confusion matrices
- prediction files
- Prediction SHA checksums

`Prediction SHA` is only an implementation-level diagnostic checksum. It is not a model component and not a performance metric.

## Evaluation Metrics

The main metrics are:

- **Accuracy**: the proportion of correctly classified validation clips.
- **Macro-F1**: the unweighted mean of per-class F1 scores. This is important for imbalanced classroom activity categories.

The paper reports both temporal split and purged temporal split results. The purged temporal split is used as a robustness diagnostic rather than as a replacement for the main temporal split.

## Reproducibility Notes

1. Use the same dataset version and directory structure.
2. Keep the same split mode when comparing results.
3. Do not reuse checkpoints across different backbone settings.
4. Check the Prediction SHA values when comparing models to ensure that different runs actually produce different prediction sequences.
5. Report both mean and standard deviation across folds.

## Citation

The citation entry will be added after publication.
