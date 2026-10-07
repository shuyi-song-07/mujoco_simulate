#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
# Activate mujoco_repro first. Pass --dataset-root, --output-dir, --steps, etc.
exec python training/ACT/train_act_dual_arm.py --device cuda "$@"
