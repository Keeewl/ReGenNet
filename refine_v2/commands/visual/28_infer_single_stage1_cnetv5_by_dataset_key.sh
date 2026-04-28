conda activate regennet5090

####### single-sample Stage1 infer/export: CNetV5 baseline #######
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=${MODEL_PATH:-PATH/TO/CNETV5_MODEL.pt}
DATASET_KEY=${DATASET_KEY:-G038T003A016R005}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/single_stage1_cnetv5_${DATASET_KEY}}

python -m sample.infer_single_stage1_clip \
  --model_path "${MODEL_PATH}" \
  --dataset interx \
  --dataset_key "${DATASET_KEY}" \
  --output_dir "${OUTPUT_DIR}" \
  --shape_mode restored_shape_height \
  --restoration_meta_path dataset/interx/cache/interx_restoration_meta.npz \
  --raw_motions_root dataset/interx/motions
