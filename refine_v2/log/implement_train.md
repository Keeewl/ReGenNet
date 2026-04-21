你现在在 ReGenNet 仓库里实现 `refine_v2` 的下一阶段：

**模块 3：stage2_refine_v2 第一版训练框架**

这轮不要再改 selector/window/subset/data interface。

这轮目标不是一次做最终版 refiner，而是把第一版可训练闭环稳定搭起来：

1. minimal mesh-aware residual refiner backbone
2. cheap but meaningful mesh/contact-aware loss
3. train / val / eval loop
4. checkpoint / logging
5. small overfit test
6. 基础命令脚本

核心验收标准：

```text
64 windows small-overfit loss 能明显下降
train/val loop 能稳定跑
eval 同时报告 coarse baseline 与 refined prediction
第一版不被 dynamic SMPL-X geometry forward 卡住
```

==================================================
一、当前前提（必须对齐）
==================================================

当前前置模块已经基本固定，不要 redesign。

### 1. selector/window 已冻结

```text
proposal_type = hand_time_with_region_attribution
selector_tau_contact = 0.10
gap_merge = 4
raw_L_min = 2
window_size = 30
per_hand_max_windows = 2
per_seq_max_windows = 3
top_k_regions = 3
```

### 2. subset 已冻结

```text
action subset = 15 selected contact-rich action types
training bucket = GT+ / Pred+
subset selector windows = refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz
```

### 3. 模块 2 数据接口已就位

已经有稳定的 fast-path refiner data interface：

```text
one window = one sample
actor/coarse/gt motion windows
coarse mesh-region contact condition
GT mesh-region supervision
primary + top-k region condition
restored-space strict alignment
dataloader / collate / inspection CLI
```

### 4. 总原则

```text
motion 是主对象
mesh-aware 信息是条件、监督和评估载体
```

也就是：

- 主输入 / 主输出：SMPL-X motion 参数链，默认沿用 dataset 中的 `[J, F, T]`
- 网络主任务：window-level residual refinement
- mesh/contact：显式进入 condition、loss、eval
- 不做 all-mesh token 模型
- 不做纯 joint-only / 纯 reconstruction baseline

==================================================
二、这轮不允许改的东西
==================================================

不要改：

1. selector/window 配置
2. subset 规则
3. GT label 定义
4. 模块 2 dataset schema
5. restored-space artifact 生成逻辑
6. one-window-with-top-k-region 的设计
7. 不要把 hand-time window 复制成多个 region-window 样本
8. 不要引入 full mesh graph transformer / all-mesh token 这类过重设计
9. 不要让第一版训练强依赖 dynamic SMPL-X geometry forward

==================================================
三、这轮实现边界
==================================================

这轮必须实现：

- backbone
- loss
- train loop
- val / eval loop
- checkpoint / logging
- small overfit test
- 基础命令脚本

这轮可以预留但不要默认强依赖：

- dynamic SMPL-X geometry forward
- predicted mesh min-distance loss
- slow geometry eval

第一版默认训练应该只依赖模块 2 fast-path batch 字段。

==================================================
四、backbone 设计要求
==================================================

##################################################
4.1 backbone 总体定位
##################################################

请实现：

```text
RefineV2WindowRefiner
```

定位：

```text
window-level residual refiner with mesh-aware conditioning
```

输出：

```text
pred_delta_motion_window
pred_motion_window = coarse_motion_window + pred_delta_motion_window
```

要求：

- 不是简单 MLP
- 不是 all-mesh transformer
- 使用时间维为主轴
- 显式接收 actor context
- 显式接收 mesh-aware condition
- 输出和 `coarse_motion_window` 同 shape

##################################################
4.2 推荐第一版结构
##################################################

输入：

```text
coarse_motion_window                 [B, J, F, T]
actor_motion_window                  [B, J, F, T]
hand_side_id                         [B]
primary_target_region_id             [B]
topk_target_region_ids               [B, K]
topk_region_scores_numeric           [B, K, 3]
coarse_region_contact_mask_window    [B, 6, T]
coarse_min_region_dist_window        [B, 6, T]
valid_mask                           [B, T]
```

建议结构：

### Step 1. Motion tokenization

将 motion flatten 成 time-major tokens：

```text
coarse_motion_window -> coarse_tokens [B, T, D]
actor_motion_window  -> actor_tokens  [B, T, D]
```

先沿用原始 `[J, F]` flatten，不要在第一版里做复杂 motion representation conversion。

### Step 2. Mesh-aware condition encoder

实现一个轻量 condition encoder，例如：

```text
hand embedding
primary region embedding
top-k region embedding + pooling
top-k score MLP
coarse contact/dist per-frame encoder
```

输出：

```text
global_condition [B, D]
per_frame_condition [B, T, D]
```

注意：

mesh/contact 条件必须进入模型 forward，不能只作为 metadata 或 loss 使用。

### Step 3. Refiner blocks

第一版建议每个 block 包含：

```text
temporal self-attention on coarse tokens
actor-conditioned cross-attention
condition additive / FiLM modulation
FFN
```

可以预留 local/spatial mixing，但不要让它成为第一版复杂度主因。

### Step 4. Output head

输出：

```text
pred_delta_tokens [B, T, J*F]
pred_delta_motion_window [B, J, F, T]
pred_motion_window = coarse_motion_window + pred_delta_motion_window
```

##################################################
4.3 不要做的设计
##################################################

这轮不要：

- full-body all-vertex token transformer
- 大型 graph mesh network
- 一个 window 复制成多个 region-conditioned forward
- 多个 experimental backbone branch 同时上线
- 第一版强依赖 geometry forward 才能训练

==================================================
五、loss 设计要求
==================================================

##################################################
5.1 总原则
##################################################

loss 不能只是普通 reconstruction。

第一版必须至少实现：

```text
L_motion
L_contact_weighted_motion
L_smooth
```

可选预留：

```text
L_region_dist
```

但 `L_region_dist` 第一版默认可以关闭。

推荐默认权重：

```text
lambda_motion = 1.0
lambda_contact = 1.0
lambda_smooth = 0.05
lambda_region_dist = 0.0
```

##################################################
5.2 motion reconstruction loss
##################################################

实现：

```text
L_motion = SmoothL1(pred_motion_window, gt_motion_window)
```

要求：

- 使用 `valid_mask`
- 输出 coarse baseline motion error，便于比较
- 如果暂时无法可靠区分 rot/trans channel，不要硬写错误假设
- 可以先支持统一权重，再预留 channel/joint weights

如果你能从现有 reaction_data schema 中可靠确认 transl/root channel，再加单独权重；否则第一版不要猜。

##################################################
5.3 contact-weighted motion loss（默认开启，必须做）
##################################################

必须使用：

```text
gt_region_contact_mask_window
coarse_region_contact_mask_window
topk_target_region_ids
```

第一版推荐做法：

```text
gt_contact_frame_mask[t] = any(gt_region_contact_mask_window[:, t] > 0)
```

然后对 contact frames 提高 motion reconstruction 权重：

```text
frame_weight[t] = 1 + contact_frame_weight * gt_contact_frame_mask[t]
```

推荐默认：

```text
contact_frame_weight = 2.0
```

可选增强：

- top-k region 中如果有 GT active region，再增加权重
- coarse predicted contact frames 也可作为较低权重 condition consistency

但不要第一版做得过复杂。

输出：

```text
loss_contact_weighted
contact_frame_ratio
coarse_contact_frame_ratio
```

##################################################
5.4 temporal smoothness loss
##################################################

实现：

```text
L_smooth = mean(abs(delta[:, :, :, 1:] - delta[:, :, :, :-1]))
```

优先对 `pred_delta_motion_window` 做一阶 smoothness。

目的：

- 减少局部 jitter
- 避免 refiner 输出大幅高频残差

##################################################
5.5 optional geometry / region-distance loss
##################################################

请预留接口，但不要让它阻塞第一版训练。

可以新建：

```text
refine_v2/train/geometry_loss.py
```

或：

```text
refine_v2/model/geometry.py
```

要求：

- dataset `__getitem__` 中绝不跑 SMPL-X
- geometry forward 只在 train/eval 中显式启用
- 默认关闭：

```text
lambda_region_dist = 0.0
enable_geometry_loss = False
```

如果实现了可选 geometry loss，建议最小形式：

```text
L_region_dist = |pred_min_region_dist - gt_min_region_dist|
```

或 improvement style：

```text
L_improve = relu(pred_dist - coarse_dist + margin)
```

仅在 GT contact frames / top-k regions 上加权。

注意：

如果 geometry helper 依赖 betas/gender/body model metadata，而模块 2 batch 暂时没有这些字段，
不要偷偷绕开 dataset schema 读别的数据。请清晰报错或保持该 loss disabled。

##################################################
5.6 loss 汇总
##################################################

实现：

```text
RefineV2Loss
```

至少输出：

```text
loss_total
loss_motion
loss_contact_weighted
loss_smooth
loss_region_dist          # 如果未启用则为 0
coarse_motion_error
pred_motion_error
contact_frame_motion_error
```

==================================================
六、train / eval 框架
==================================================

##################################################
6.1 train loop
##################################################

请实现最小但完整的 train loop，例如：

```text
refine_v2/train/trainer.py
refine_v2/cli_train_refiner.py
```

功能至少包括：

- 读取模块 2 dataset/loader
- deterministic train/val split
- optimizer
- lr scheduler
- grad clip
- checkpoint save/load
- log step / epoch summary
- best checkpoint by chosen val metric
- resume from checkpoint

重要：

train/val split 必须按 sequence / `dataset_row_index` 划分，不要按 window 随机划分。

原因：

同一 sequence 可能有多个 windows。window-level random split 会让同一序列同时出现在 train 和 val，导致 val 偏乐观。

推荐默认：

```text
val_ratio = 0.1
split_seed = 1234
split unit = dataset_row_index
```

##################################################
6.2 val / eval loop
##################################################

这轮先做 window-level eval，不做 full-sequence stitching。

至少实现：

### A. reconstruction

```text
coarse_motion_error
pred_motion_error
motion_improvement = coarse_motion_error - pred_motion_error
```

### B. contact-aware motion eval

在 GT contact frames 上比较：

```text
coarse_contact_motion_error
pred_contact_motion_error
contact_motion_improvement
```

### C. metadata breakdown

至少按以下维度做 breakdown：

```text
action_type
hand_side
primary_target_region
```

### D. optional geometry eval

如果启用 geometry helper，再报告：

```text
coarse_region_dist_error
pred_region_dist_error
region_dist_improvement
```

但这不是第一版默认必须项。

##################################################
6.3 small overfit test（必须做）
##################################################

必须实现 overfit 模式：

```text
--overfit_num_windows 64
--num_steps 500
```

要求：

- 只训练固定小窗口集合
- 可以关闭 val 或把同一小集合作为 sanity eval
- 打印初始 loss 和最终 loss
- 如果 loss 没明显下降，最终报告要明确提示

这个模式是第一版训练框架的主要验收手段。

==================================================
七、建议目录
==================================================

请优先在这些位置实现（可微调，但不要乱拆）：

```text
refine_v2/model/refiner_v2.py
refine_v2/model/condition_encoder.py
refine_v2/model/losses_v2.py
refine_v2/train/trainer.py
refine_v2/train/eval_window.py
refine_v2/train/geometry_loss.py      # optional / default disabled
refine_v2/cli_train_refiner.py
refine_v2/cli_eval_refiner.py
refine_v2/commands/12_train_refiner.sh
refine_v2/commands/13_eval_refiner.sh
```

可以补：

```text
refine_v2/model/README.md
refine_v2/train/README.md
```

==================================================
八、训练输入字段（必须对齐模块 2）
==================================================

模型 forward / loss 必须使用模块 2 batch dict。

至少接收：

### motion 主干

```text
actor_motion_window
coarse_motion_window
gt_motion_window
```

### mesh-aware 条件

```text
hand_side_id
primary_target_region_id
topk_target_region_ids
topk_region_scores_numeric
coarse_region_contact_mask_window
coarse_min_region_dist_window
```

### 监督 / mask

```text
gt_region_contact_mask_window
gt_min_region_dist_window
valid_mask
```

不要绕开这些字段单独自己再读 raw artifact。

==================================================
九、最小运行命令
==================================================

### 1. small overfit test

```bash
python -m refine_v2.cli_train_refiner \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --save_dir refine_v2/outputs/train/refiner_v2_overfit \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda \
  --overfit_num_windows 64 \
  --num_steps 500 \
  --lambda_region_dist 0.0
```

### 2. full subset train

```bash
python -m refine_v2.cli_train_refiner \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --save_dir refine_v2/outputs/train/refiner_v2_exp1 \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  --val_ratio 0.1 \
  --split_seed 1234 \
  --lambda_region_dist 0.0
```

### 3. eval

```bash
python -m refine_v2.cli_eval_refiner \
  --checkpoint refine_v2/outputs/train/refiner_v2_exp1/model_best.pt \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --device cuda
```

==================================================
十、实现风格要求
==================================================

1. 不改 selector/window/subset/data interface
2. 不把模型退化成 pure reconstruction baseline
3. 不做 all-mesh 大模型
4. 第一版优先保证 overfit 和 train loop 稳定
5. mesh/contact 必须进入 model condition 和 loss
6. geometry loss 预留接口，默认关闭或权重为 0
7. eval 必须报告 coarse baseline vs refined prediction
8. train/val 必须 sequence-level split
9. 日志要清楚，能直接看 loss 是否下降、refined 是否优于 coarse
10. checkpoint 要能 resume

==================================================
十一、实现完成后必须返回
==================================================

请返回：

1. 修改/新增文件列表
2. 每个文件职责
3. backbone 结构说明
4. mesh-aware condition 是如何进入模型的
5. loss 结构说明
6. geometry/region-distance loss 的当前状态：默认关闭、可选、或已实现
7. train/eval loop 说明
8. sequence-level split 说明
9. overfit test 是否能正常跑，loss 是否下降
10. 最小运行命令
11. 第一版 refiner 的预期和后续最值得调的点

先直接实现，不要只停留在分析。
