#!/usr/bin/env bash
set -euo pipefail

# Print the stored fields and shapes from reaction_data.npz.
#
# Example:
#   REACTION_DATA_PATH=tmp/refine/interx/train/reaction_data.npz \
#   bash refine/commands/04_show_reaction_data_fields.sh

REACTION_DATA_PATH="${REACTION_DATA_PATH:?Set REACTION_DATA_PATH to reaction_data.npz}"

python3 - <<'PY'
import os
import numpy as np

path = os.environ["REACTION_DATA_PATH"]
data = np.load(path, allow_pickle=True)
print("fields=", sorted(data.files))
for key in sorted(data.files):
    value = data[key]
    print(f"{key}: shape={value.shape}, dtype={value.dtype}")
PY
