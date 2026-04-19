conda activate regennet5090

####### Stage2: eval stgcn #######
export CUDA_VISIBLE_DEVICES=7
python3 -m refine.eval.global_motion \
  --pack refine/outputs/eval_stage2_lite_step000019000_test1000/refined_pack.npz \
  --dataset interx \
  --stgcn_model_path recognition_training/interx_exp1/checkpoint_0100.pth.tar \
  --body_model smplx \
  --batch_size 64 \
  --device cuda:0 \
  --json_out refine/outputs/eval_stage2_lite_step000019000_test1000/global_eval.json \
  --csv_out refine/outputs/eval_stage2_lite_step000019000_test1000/global_eval.csv
