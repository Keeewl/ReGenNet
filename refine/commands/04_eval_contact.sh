conda activate regennet5090

####### Stage2: eval contact #######
export CUDA_VISIBLE_DEVICES=7
python3 -m refine.eval.local_contact \
  --pack refine/outputs/eval_stage2_lite_step000019000_test1000/refined_pack.npz \
  --device cuda:0 \
  --body_model smplx \
  --pose_rep rot6d \
  --batch_size 64 \
  --json_out refine/outputs/eval_stage2_lite_step000019000_test1000/local_eval.json \
  --csv_out refine/outputs/eval_stage2_lite_step000019000_test1000/local_eval.csv

####### Stage2: eval contact (subset) #######
export CUDA_VISIBLE_DEVICES=6
python -m refine.eval.local_contact \
  --pack refine/outputs/contact_eval_step19000_train/contact_refined_pack.npz \
  --device cuda:0 \
  --body_model smplx \
  --pose_rep rot6d \
  --batch_size 64 \
  --json_out refine/outputs/contact_eval_step19000_train/local_eval.json \
  --csv_out refine/outputs/contact_eval_step19000_train/local_eval.csv
