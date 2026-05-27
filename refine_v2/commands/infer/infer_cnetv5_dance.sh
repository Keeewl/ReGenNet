conda activate regennet5090

####### single-sample Stage1 infer/export: CNetV5 baseline #######
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH=save/cnet_v5_256/interx_smplx_online_exp1/model000209455.pt
DATASET_KEY=G027T004A021R004
OUTPUT_DIR=outputs/single_stage1_cnetv5_G027T004A021R004

python -m sample.infer_single_stage1_clip \
  --model_path "${MODEL_PATH}" \
  --dataset interx \
  --dataset_key "${DATASET_KEY}" \
  --output_dir "${OUTPUT_DIR}" \
  --shape_mode restored_shape_height \
  --restoration_meta_path dataset/interx/cache/interx_restoration_meta.npz \
  --raw_motions_root dataset/interx/motions \
  --use_ddim \
  --timestep_respacing ddim5
