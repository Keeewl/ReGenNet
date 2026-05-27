conda activate regennet5090

####### single-sample Stage1 infer/export: CMDM baseline #######
export CUDA_VISIBLE_DEVICES=3

MODEL_PATH=${MODEL_PATH:-PATH/TO/CMDM_MODEL.pt}
DATASET_KEY="G053T000A000R004"
OUTPUT_DIR="single_stage1_cmdm_G053T000A000R004"

python -m sample.infer_single_stage1_clip \
  --model_path save/cmdm/interx_smplx_online_exp1/model000149455.pt \
  --dataset interx \
  --dataset_key "${DATASET_KEY}" \
  --output_dir "${OUTPUT_DIR}" \
  --shape_mode restored_shape_height \
  --restoration_meta_path dataset/interx/cache/interx_restoration_meta.npz \
  --raw_motions_root dataset/interx/motions \
  --use_ddim \
  --timestep_respacing ddim5