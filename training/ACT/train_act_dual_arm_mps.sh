#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec python training/ACT/train_act_dual_arm.py --device mps "$@"
