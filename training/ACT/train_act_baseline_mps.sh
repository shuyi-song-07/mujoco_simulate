#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${0:A:h:h:h}"
DATASET_ROOT="${DATASET_ROOT:-$PROJECT_ROOT/datasets/act_50eps}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/act_panda_baseline_official}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin}"

if [[ ! -d "$DATASET_ROOT/meta" ]]; then
  echo "Dataset not found at: $DATASET_ROOT"
  echo "Set DATASET_ROOT to the downloaded LeRobot dataset directory."
  exit 1
fi

HF_HOME="$PROJECT_ROOT/.cache/huggingface" \
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
exec "$PYTHON_BIN/lerobot-train" \
  --dataset.repo_id=shuyisong07/act_50eps \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --dataset.eval_split=0.1 \
  --dataset.use_imagenet_stats=true \
  --policy.type=act \
  --policy.device=mps \
  --policy.vision_backbone=resnet18 \
  --policy.pretrained_backbone_weights=ResNet18_Weights.IMAGENET1K_V1 \
  --policy.chunk_size=100 \
  --policy.n_action_steps=100 \
  --policy.dim_model=512 \
  --policy.n_heads=8 \
  --policy.dim_feedforward=3200 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.use_vae=true \
  --policy.latent_dim=32 \
  --policy.n_vae_encoder_layers=4 \
  --policy.kl_weight=10 \
  --policy.optimizer_lr=0.00001 \
  --policy.optimizer_lr_backbone=0.00001 \
  --policy.optimizer_weight_decay=0.0001 \
  --policy.push_to_hub=false \
  --seed=1000 \
  --batch_size=8 \
  --num_workers=0 \
  --persistent_workers=false \
  --steps=100000 \
  --eval_steps=5000 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --log_freq=100 \
  --env_eval_freq=0 \
  --output_dir="$OUTPUT_DIR" \
  --job_name=act_panda_baseline_official \
  --wandb.enable=false
