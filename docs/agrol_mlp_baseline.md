# AGRoL MLP Baseline (Stage1 CMDM + MLP)

This note documents the minimal commands for offline and strict-online (sliding-window) baselines.

## Offline (full-sequence)

Train:
```
mpiexec -n 4 --allow-run-as-root \
  python -m train.train_mdm \
  --setting cmdm --arch mlp --cm_mode concat \
  --save_dir save/cmdm/chi3d_smplx_mlp_offline \
  --dataset chi3d --num_person 2 --num_frames 150 \
  --pose_rep rot6d --body_model smplx \
  --data_path PATH/TO/chi3d_smplx_train.h5 \
  --cond_mask_prob 0 --train_platform_type TensorboardPlatform --overwrite
```

Eval:
```
python -m eval.eval_cmdm \
  --model PATH/TO/model_XXXX.pt \
  --rec_model_path PATH/TO/checkpoint_0100.pth.tar \
  --eval_mode full --use_ddim --timestep_respacing ddim5
```

## Online (sliding-window)

Train (strict online, windowed supervision):
```
mpiexec -n 4 --allow-run-as-root \
  python -m train.train_mdm \
  --setting cmdm --arch mlp --cm_mode concat \
  --save_dir save/cmdm/chi3d_smplx_mlp_online \
  --dataset chi3d --num_person 2 --num_frames 150 \
  --pose_rep rot6d --body_model smplx \
  --data_path PATH/TO/chi3d_smplx_train.h5 \
  --cond_mask_prob 0 --train_platform_type TensorboardPlatform --overwrite \
  --reaction_mode online --online_strategy sliding_window \
  --window_size 30 --window_stride 10 --window_emit stride --window_pad_mode edge \
  --online_train_random_offset
```

Eval (online sliding-window generation):
```
python -m eval.eval_cmdm \
  --model PATH/TO/model_XXXX.pt \
  --rec_model_path PATH/TO/checkpoint_0100.pth.tar \
  --eval_mode full --use_ddim --timestep_respacing ddim5 \
  --reaction_mode online --online_strategy sliding_window \
  --window_size 30 --window_stride 10 --window_emit stride --window_pad_mode edge
```

Notes:
- `--reaction_mode online` switches training/eval to strict online sliding-window.
- `--window_emit stride` supervises/emits only the last `window_stride` frames of each window.
