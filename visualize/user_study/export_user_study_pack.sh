#!/usr/bin/env bash
set -euo pipefail

# Batch-export viewer-ready assets for user study clips.
#
# For each dataset_key in DATASET_KEYS_FILE, this script exports:
#   1) GT clip copied from GT_DATA_DIR
#   2) baseline Stage1 output (typically CMDM / ReGenNet baseline shell)
#   3) HiReact Stage1 output
#   4) HiReact Stage2 refined/coarse/gt variants
#
# Output layout:
#   OUTPUT_ROOT/<dataset_key>/
#     gt/<dataset_key>/P1.npz,P2.npz
#     baseline/...               # full single-stage1 export dir
#     stage1/...                 # full single-stage1 export dir
#     stage2/coarse/<dataset_key>/...
#     stage2/refined/<dataset_key>/...
#     stage2/gt/<dataset_key>/...
#
# Required environment variables:
#   BASELINE_MODEL_PATH   path to baseline Stage1 checkpoint
#   STAGE1_MODEL_PATH     path to HiReact Stage1 checkpoint
#   STAGE2_CHECKPOINT     path to HiReact Stage2 refiner checkpoint
#
# Optional environment variables:
#   DATASET_KEYS_FILE     text file with one dataset_key per line
#   OUTPUT_ROOT           root output directory
#   GT_DATA_DIR           GT viewer-ready clip root
#   RAW_MOTIONS_ROOT      Inter-X raw motions root used by single clip export
#   RESTORATION_META_PATH Inter-X restoration metadata package
#   REGION_MAP_PATH       region map for Stage2
#   CUDA_VISIBLE_DEVICES  exported externally when needed

DATASET_KEYS_FILE=${DATASET_KEYS_FILE:-visualize/user_study/dataset_keys.txt}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/user_study_pack}
GT_DATA_DIR=${GT_DATA_DIR:-outputs/interx_regen_train_restored_height}
RAW_MOTIONS_ROOT=${RAW_MOTIONS_ROOT:-dataset/interx/motions}
RESTORATION_META_PATH=${RESTORATION_META_PATH:-dataset/interx/cache/interx_restoration_meta.npz}
REGION_MAP_PATH=${REGION_MAP_PATH:-visualize/viewer/part_segm/6_parts/six_parts.pkl}

BASELINE_MODEL_PATH=${BASELINE_MODEL_PATH:-}
STAGE1_MODEL_PATH=${STAGE1_MODEL_PATH:-}
STAGE2_CHECKPOINT=${STAGE2_CHECKPOINT:-}

if [[ -z "${BASELINE_MODEL_PATH}" ]]; then
  echo "BASELINE_MODEL_PATH is required" >&2
  exit 1
fi
if [[ -z "${STAGE1_MODEL_PATH}" ]]; then
  echo "STAGE1_MODEL_PATH is required" >&2
  exit 1
fi
if [[ -z "${STAGE2_CHECKPOINT}" ]]; then
  echo "STAGE2_CHECKPOINT is required" >&2
  exit 1
fi
if [[ ! -f "${DATASET_KEYS_FILE}" ]]; then
  echo "DATASET_KEYS_FILE not found: ${DATASET_KEYS_FILE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

while IFS= read -r raw_line; do
  dataset_key=$(printf '%s' "${raw_line}" | sed 's/#.*$//' | xargs)
  if [[ -z "${dataset_key}" ]]; then
    continue
  fi

  echo "=== [${dataset_key}] export user-study pack ==="
  key_root="${OUTPUT_ROOT}/${dataset_key}"
  mkdir -p "${key_root}"

  gt_src="${GT_DATA_DIR}/${dataset_key}"
  gt_dst_root="${key_root}/gt"
  mkdir -p "${gt_dst_root}"
  if [[ ! -d "${gt_src}" ]]; then
    echo "GT clip not found: ${gt_src}" >&2
    exit 1
  fi
  rm -rf "${gt_dst_root:?}/${dataset_key}"
  cp -R "${gt_src}" "${gt_dst_root}/"

  baseline_out="${key_root}/baseline"
  python -m sample.infer_single_stage1_clip \
    --model_path "${BASELINE_MODEL_PATH}" \
    --dataset interx \
    --dataset_key "${dataset_key}" \
    --output_dir "${baseline_out}" \
    --shape_mode restored_shape_height \
    --restoration_meta_path "${RESTORATION_META_PATH}" \
    --raw_motions_root "${RAW_MOTIONS_ROOT}"

  stage1_out="${key_root}/stage1"
  python -m sample.infer_single_stage1_clip \
    --model_path "${STAGE1_MODEL_PATH}" \
    --dataset interx \
    --dataset_key "${dataset_key}" \
    --output_dir "${stage1_out}" \
    --shape_mode restored_shape_height \
    --restoration_meta_path "${RESTORATION_META_PATH}" \
    --raw_motions_root "${RAW_MOTIONS_ROOT}"

  stage2_out="${key_root}/stage2"
  python -m refine_v2.cli_infer_refiner_on_viewer_clip \
    --checkpoint "${STAGE2_CHECKPOINT}" \
    --dataset interx \
    --data_dir "${stage1_out}/motions" \
    --clip_name "${dataset_key}" \
    --region_map_path "${REGION_MAP_PATH}" \
    --output_dir "${stage2_out}" \
    --variant all \
    --device cuda \
    --tau_contact 0.05 \
    --gap_merge 4 \
    --raw_L_min 12 \
    --window_size 30 \
    --per_hand_max_windows 2 \
    --per_seq_max_windows 4 \
    --top_k_regions 3 \
    --frame_chunk 1 \
    --target_chunk 2048

done < "${DATASET_KEYS_FILE}"

echo "saved user-study export pack: ${OUTPUT_ROOT}"
