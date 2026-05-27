conda activate regennet5090

####### single-sample Stage1 infer/export: CMDM baseline #######
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=${MODEL_PATH:-PATH/TO/CMDM_MODEL.pt}
DATASET_KEY=${DATASET_KEY:-G038T003A016R005}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/single_stage1_cmdm_${DATASET_KEY}}

python -m sample.infer_single_stage1_clip \
  --model_path save/cmdm/interx_smplx_online_exp1/model000199455.pt \
  --dataset interx \
  --dataset_key "${DATASET_KEY}" \
  --output_dir "${OUTPUT_DIR}" \
  --shape_mode restored_shape_height \
  --restoration_meta_path dataset/interx/cache/interx_restoration_meta.npz \
  --raw_motions_root dataset/interx/motions \
  --use_ddim \
  --timestep_respacing ddim5