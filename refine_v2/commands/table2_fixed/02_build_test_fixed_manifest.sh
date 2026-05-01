#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m refine_v2.cli_build_fixed_eval_manifest \
  --reaction_data_path refine_v2/save/table2_test/mainchain_test/reaction_data.npz \
  --selected_action_types \
    A028 A025 A001 A009 A021 A000 A008 A019 A023 A035 A027 A022 A003 A016 A034 \
  --bucket_label FIXED \
  --output_dir refine_v2/save/table2_fixed/test
