你现在在 ReGenNet 仓库里实现 `refine_v2` 的下一阶段：

**模块 2：refiner data interface + fast window feature dataset**

这轮不要实现网络，不要实现 loss，不要训练。

只实现一个最小、稳定、可检查的 refiner 数据接口：

1. subset-window 数据读取与严格对齐
2. restored-space 一致性检查
3. motion 主干 window crop
4. 直接 slice 已有 mesh-region contact condition / supervision
5. dataloader / collate
6. inspection / sanity-check CLI

重要：这轮的目标是 **给下一轮 minimal residual refiner 提供可靠数据接口**，不是把 feature 工程一次做满。

==================================================
一、当前前提（必须对齐）
==================================================

当前 module 1 和 contact-rich subset 已基本固定，不要 redesign。

### 已固定的 selector/window 配置

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

### 已固定的 subset 策略

```text
action subset = 15 selected action types
training bucket = GT+ / Pred+
selector/window = frozen hand-time top-k tau010
```

### 已有 artifacts

```text
refine/dataset/train/reaction_data.npz
refine_v2/outputs/train/contact_labels_gt.npz
refine_v2/outputs/train/contact_subset/subset_manifest.json
refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz
refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_audit.json
refine_v2/outputs/train/contact_subset/selector_rerun/subset_window_metadata.json
```

### 当前模块 2 的设计原则

```text
motion 是主对象
mesh/contact 是条件、监督和评估载体
```

也就是：

- 主输入 / 主输出：当前 SMPL-X motion 参数链，默认 rot6d(+trans)
- 条件：hand / region / top-k / coarse contact mask-distance
- 监督：GT motion window + GT mesh-region contact mask-distance

==================================================
二、这轮不允许改的东西
==================================================

不要改：

1. selector/window 配置
2. subset 规则
3. GT label 定义
4. contact mask / min-distance 定义
5. restored-space artifact 生成逻辑
6. feature/network/loss/train
7. one-window-with-top-k-region 的设计
8. 不要把一个 hand-time window 复制成多个 region-window 样本
9. 不要退回 v1 的 joint-only feature 方案

==================================================
三、这轮核心实现原则（重要）
==================================================

### 原则 1：默认走 fast path

Dataset 默认不要重新跑 SMPL-X，不要重新算 mesh distance。

必须优先直接 slice 现有 artifacts：

```text
reaction_data.npz                         -> motion crop
subset_selector_windows.npz               -> coarse pred contact mask / min dist crop
contact_labels_gt.npz                     -> GT contact mask / min dist crop
subset_manifest.json / window_metadata    -> metadata / action / bucket / top-k
```

### 原则 2：xyz 只作为 optional debug，不作为默认输出

可以预留 `include_xyz=False` 参数。

默认：

```text
不输出 actor_xyz_window / coarse_xyz_window / gt_xyz_window
不在 __getitem__ 内跑 SMPL-X forward
```

如果实现 `include_xyz=True`，只作为 debug/inspection 辅助字段，不作为训练 fast path 的默认字段。

### 原则 3：refiner dataset 阶段只做 restored-space 一致性检查

当前 subset/window/contact artifacts 已经由 restored-space pipeline 生成。

因此 dataset 默认要求：

```text
reaction_data
contact_labels
selector_windows
```

都声明 / 对齐到 `restored_pair_space`。

如果不一致：

```text
明确报错
不要在 dataset 里静默自动 restore
```

原因：dataset 阶段隐式 restore motion，可能导致 motion 和已生成 selector/contact artifact 空间错位。

### 原则 4：row mapping 必须显式处理

注意：`subset_selector_windows.npz` 是 subset rerun artifact，它内部数组是 subset-local 顺序。

不能直接用 `dataset_row_index` 去索引 selector arrays。

必须建立：

```text
reaction_data row index = dataset_row_index
label_row_to_index      = {dataset_row_index -> label array index}
selector_row_to_index   = {dataset_row_index -> selector artifact local index}
manifest_row_to_record  = {dataset_row_index -> manifest sequence metadata}
```

所有 motion/contact/mask/window 必须通过这些 mapping 对齐。

==================================================
四、样本单位与输出 schema
==================================================

##################################################
4.1 样本单位：一窗一条样本
##################################################

每个训练样本对应：

```text
subset 中的一条 sequence
selector rerun 选出的一个 hand-time window
```

重要：

```text
不要复制成多个 region-window 样本
一个 window 样本 = motion crop + hand + primary region + top-k regions + contact supervision
```

##################################################
4.2 每个 window sample 的最小输出字段
##################################################

### A. motion 主干

必须输出：

```text
actor_motion_window      # [J, F, T]
coarse_motion_window     # [J, F, T]
gt_motion_window         # [J, F, T]
```

默认保持 reaction_data 中的 motion 表示，优先 rot6d(+trans)。

### B. mesh-aware condition，来自 selector artifact

必须输出当前 selected hand 对 6 个 target regions 的 coarse condition：

```text
coarse_region_contact_mask_window   # [6, T]
coarse_min_region_dist_window       # [6, T]
```

来源：

```text
subset_selector_windows.npz / pred_contact_mask
subset_selector_windows.npz / pred_min_region_dist
```

slice 方式：

```text
selector_local_index = selector_row_to_index[dataset_row_index]
hand_id = window.hand_side_id
start:end = window bounds
pred_contact_mask[selector_local_index, hand_id, :, start:end]
pred_min_region_dist[selector_local_index, hand_id, :, start:end]
```

如果 artifact 里已有 `hand_contact_mask` / `hand_min_dist`，可以额外输出，但不是最小必需。

### C. GT mesh-region supervision，来自 GT labels artifact

必须输出当前 selected hand 对 6 个 target regions 的 GT supervision：

```text
gt_region_contact_mask_window       # [6, T]
gt_min_region_dist_window           # [6, T]
```

来源：

```text
contact_labels_gt.npz / gt_contact_mask
contact_labels_gt.npz / gt_min_region_dist
```

slice 方式：

```text
label_index = label_row_to_index[dataset_row_index]
hand_id = window.hand_side_id
start:end = window bounds
gt_contact_mask[label_index, hand_id, :, start:end]
gt_min_region_dist[label_index, hand_id, :, start:end]
```

### D. hand / region / top-k condition

必须输出：

```text
hand_side
hand_side_id
primary_target_region
primary_target_region_id
topk_target_regions
topk_target_region_ids       # [K]
topk_region_scores           # 原始 list[dict] 保留到 metadata
```

同时建议提取一个数值 tensor：

```text
topk_region_scores_numeric   # [K, 3]
```

其中 3 个 score 是：

```text
num_contact_frames
mean_min_dist
min_dist
```

如果某个 score 缺字段，明确报错或填合理默认并 warning；不要静默吞掉结构错误。

### E. mask / frame / meta

必须输出：

```text
valid_mask             # [T]
window_length
start_frame
end_frame
raw_start_frame
raw_end_frame
dataset_row_index
sample_index
dataset_key
action_type
action_label
action_name
bucket_label
is_gt_positive
is_pred_positive
window_index
sequence_window_index
```

### F. optional debug fields

可选但不要默认强制：

```text
actor_xyz_window
coarse_xyz_window
gt_xyz_window
region_score_table
coarse_region_contact_summary
gt_region_contact_summary
```

==================================================
五、建议实现目录
==================================================

请按当前实际仓库结构实现，不要新建 `regennet/` 路径。

优先放在：

```text
refine_v2/refiner_data/
refine_v2/tools/inspect_refiner_data.py
refine_v2/cli_inspect_refiner_data.py
```

建议结构：

```text
refine_v2/refiner_data/__init__.py
refine_v2/refiner_data/schema.py
refine_v2/refiner_data/window_dataset.py
refine_v2/refiner_data/window_loader.py
refine_v2/refiner_data/feature_pack.py
refine_v2/refiner_data/sanity_checks.py
refine_v2/refiner_data/README.md
```

可以微调，但不要拆太散。

==================================================
六、具体实现要求
==================================================

##################################################
6.1 RefineV2WindowDataset
##################################################

实现：

```text
RefineV2WindowDataset
```

支持输入：

```text
reaction_data_path
contact_labels_path
subset_manifest_path
selector_windows_path
include_buckets              default: ["GT+ / Pred+"]
selected_action_types        optional
include_xyz                  default: False
strict_checks                default: True
```

行为：

1. 读取 subset manifest
2. 只保留 `include_buckets` 中的 sequences
3. 如果传 `selected_action_types`，再按 action type 过滤
4. 读取 selector windows 中属于这些 sequences 的 windows
5. 每个 `__getitem__` 返回一个 window sample dict
6. 不静默丢弃不一致样本；strict 模式下直接报错

##################################################
6.2 feature pack builder
##################################################

实现一个明确的函数或类，把 raw window record 转成模型将来能直接吃的 dict，例如：

```text
build_window_feature_sample(...)
```

职责：

- crop actor/coarse/gt motion
- crop coarse contact mask/dist
- crop GT contact mask/dist
- 打包 hand/region/top-k 条件
- 构造 valid_mask
- 附带 metadata

不要在这里做网络专用 normalization。

##################################################
6.3 sanity checks（必须做）
##################################################

至少检查：

1. motion window 长度和 selector bounds 对齐
2. GT/coarse contact window 和 motion window 时间对齐
3. top-k region ids 合法、非空、长度一致
4. `primary_target_region_id` 在 `topk_target_region_ids` 中，若不在，明确 warning 或报错
5. valid mask shape / dtype 正确
6. `dataset_row_index` 在 reaction_data / labels / selector / manifest 中都存在
7. selector artifact local index 与 `dataset_row_indices` 对齐
8. `space_definition` / metadata 声明 restored pair space
9. motion crop 不越界，不超过 sequence length
10. contact mask/dist 的时间维和 window length 一致

如果不一致：

```text
明确报错
不要静默跳过
```

##################################################
6.4 dataloader / collate
##################################################

提供：

```text
make_refine_v2_window_loader(...)
```

支持：

```text
batch_size
shuffle
num_workers
```

实现 collate_fn：

Tensor stack：

```text
actor_motion_window
coarse_motion_window
gt_motion_window
coarse_region_contact_mask_window
coarse_min_region_dist_window
gt_region_contact_mask_window
gt_min_region_dist_window
valid_mask
hand_side_id
primary_target_region_id
topk_target_region_ids
topk_region_scores_numeric
```

Metadata 保留 list：

```text
dataset_key
action_type
action_label
action_name
bucket_label
hand_side
primary_target_region
topk_target_regions
topk_region_scores
region_score_table, if present
```

##################################################
6.5 optional xyz debug
##################################################

可以预留 `include_xyz=True`，但如果实现会拖慢或复杂化，可以先明确 `NotImplementedError`，并在 README/CLI 里说明：

```text
xyz debug will be added after the fast motion/contact dataset is stable
```

不要为了 xyz 牺牲 fast path 稳定性。

==================================================
七、inspection CLI（必须有）
==================================================

实现：

```text
python -m refine_v2.cli_inspect_refiner_data ...
```

功能：

### A. dataset summary

`--summary_only` 时打印：

```text
num_sequences
num_windows
action_type distribution
bucket distribution
hand_side distribution
primary region distribution
top-k region distribution
motion shapes
contact condition shapes
GT supervision shapes
```

### B. inspect one window sample

支持：

```text
--window_index 0
```

或：

```text
--dataset_row_index ... --start_frame ... --hand_side ...
```

输出：

```text
sequence id / action type
hand side
start/end/raw bounds
primary / top-k region
valid length
motion window shapes
coarse contact ratios by region
gt contact ratios by region
primary-region coarse/GT contact ratio
min distance summaries
```

### C. export json

支持：

```text
--output_json path.json
```

导出 summary 或单样本摘要。

不要默认导出大 tensor 到 JSON。只导出 shape、ratio、min/max/mean 等摘要。

==================================================
八、最小运行命令
==================================================

### 1. dataset summary

```bash
python -m refine_v2.cli_inspect_refiner_data \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --summary_only
```

### 2. inspect one sample

```bash
python -m refine_v2.cli_inspect_refiner_data \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --window_index 0 \
  --output_json refine_v2/outputs/train/contact_subset/refiner_data/sample0_summary.json
```

### 3. optional selected action inspect

```bash
python -m refine_v2.cli_inspect_refiner_data \
  --reaction_data_path refine/dataset/train/reaction_data.npz \
  --contact_labels_path refine_v2/outputs/train/contact_labels_gt.npz \
  --subset_manifest_path refine_v2/outputs/train/contact_subset/subset_manifest.json \
  --selector_windows_path refine_v2/outputs/train/contact_subset/selector_rerun/subset_selector_windows.npz \
  --include_buckets "GT+ / Pred+" \
  --selected_action_types Handshake \
  --summary_only
```

==================================================
九、实现风格要求
==================================================

1. 不实现网络
2. 不实现 loss
3. 不实现 train loop
4. 不改 selector/window
5. 不改 subset 规则
6. 不改 GT label 定义
7. 默认不动态算 xyz / mesh
8. 默认不重新算 contact distance
9. 只 slice 已有 artifact
10. 检查要严格，报错要清楚
11. 输出要便于人工 review
12. 保持 schema 简洁，便于下一轮 minimal residual refiner 直接复用

==================================================
十、实现完成后必须返回
==================================================

请返回：

1. 修改/新增文件列表
2. 每个文件职责
3. window-level dataset schema
4. motion 主干 + mesh-aware condition/supervision 的组织方式
5. row mapping / 对齐逻辑说明
6. sanity checks 列表
7. dataloader / collate 设计说明
8. inspection CLI 使用方式
9. 最小运行命令
10. 如果没有实现 xyz debug，明确说明原因
11. 下一轮最自然的工作：minimal residual refiner

先直接实现，不要只停留在分析。
