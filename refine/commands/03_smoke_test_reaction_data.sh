#!/usr/bin/env bash
set -euo pipefail

# Read reaction_data.npz and verify dataset + restored metadata extraction.
#
# Example:
#   REACTION_DATA_PATH=tmp/refine/interx/train/reaction_data.npz \
#   bash refine/commands/03_smoke_test_reaction_data.sh

REACTION_DATA_PATH="${REACTION_DATA_PATH:?Set REACTION_DATA_PATH to reaction_data.npz}"

python3 - <<'PY'
import os
from refine.data.cache_dataset import ReactionDataDataset
from refine.data.collate import reaction_data_collate
from refine.data.restored_space import extract_restoration_metadata

path = os.environ["REACTION_DATA_PATH"]
ds = ReactionDataDataset(path)
print("len=", len(ds))
print("item_keys=", sorted(ds[0].keys()))

batch = reaction_data_collate([ds[0], ds[min(1, len(ds) - 1)]])
print("batch_keys=", sorted(batch.keys()))

try:
    meta = extract_restoration_metadata(batch)
    print("restoration_meta_keys=", sorted(meta.keys()))
except Exception as exc:
    print("restoration_meta_status=missing_or_incomplete")
    print(type(exc).__name__, str(exc))
PY
