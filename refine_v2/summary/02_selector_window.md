# Stage2 Selector / Window 设计

## 1. Window 设计的定义、目的与整体框架

当前 `refine_v2` 中的 selector / window 不是一个学习式 proposal 模块，而是一个：

```text
deterministic local window sampler
```

它的目标不是直接预测 refined motion，而是先从 Stage1 的 coarse reactor motion 中，稳定找出：

```text
哪些局部 hand-contact 时段最值得 Stage2 去修正
```

因此，这个模块在整个 Stage2 框架中的角色是：

1. 将 coarse motion 中潜在的接触相关时段提取出来。
2. 把这些时段转成统一长度的 local windows。
3. 用尽量少的窗口覆盖尽量多的高价值接触区域。
4. 为后续 refiner 提供稳定、可重复、可审计的局部输入单元。

这个设计的核心思想是：

```text
先用确定性规则把“哪里值得修”固定下来，
再把学习能力集中到“怎么修”上。
```

因此，当前 selector / window 设计不是泛化的 sequence proposal，而是一个明确服务于 Stage2 local contact refinement 的窗口设计算法。

从实现上看，整体流程可以概括为：

```text
coarse motion
-> coarse contact representation
-> hand-time raw segment
-> region attribution
-> fixed window
-> caps / sorting
-> final window artifact
```

## 2. 基础几何量与 Coarse Contact 表示

### 2.1 基本输入

window 设计的基础输入来自 `reaction_data` 中的：

- `actor_motion`
- `reactor_coarse`
- `lengths`

也就是说，selector 工作在：

```text
actor motion + Stage1 coarse reactor motion
```

上，而不是 GT motion 上。

### 2.2 基础几何量

为了判断 coarse motion 中哪里可能存在接触，当前实现会先在 restored pair space 中，对每一帧计算：

```text
reactor left/right hand
to
actor each target region
```

的最小顶点距离。

这里使用的 target region 是 actor 的 6-part region：

- `torso_head`
- `lower_body`
- `left_arm`
- `right_arm`
- `left_hand`
- `right_hand`

因此，最基础的连续几何量是：

```text
min_region_dist[hand, region, t]
```

### 2.3 二值 coarse contact mask

有了最小距离以后，再通过阈值 `tau_contact` 将其二值化：

```text
contact_mask = (min_region_dist < tau_contact)
```

这一步得到的是 coarse contact 的二值表示：

```text
pred_contact_mask[hand, region, t]
```

因此，当前 selector 的起点不是抽象动作特征，而是一个明确的 coarse contact 表示：

```text
hand x region x time
```

### 2.4 hand-level contact 表示

由于 proposal 的第一步不是按 region 建，而是按 hand-time 建，所以实现中还会先在 region 维度上聚合：

```text
hand_contact_mask[hand, t] = any_region_contact
```

也就是说：

```text
[hand, region, t] -> [hand, t]
```

这一步很关键，因为它决定了当前 proposal 的基本单位是：

```text
sample x hand x time
```

而不是：

```text
sample x hand x region x time
```

## 3. 从 Coarse Motion 到 Hand-Time Raw Segment

### 3.1 为什么先做 hand-time segment

当前设计并不直接构造 hand-region proposal。  
原因是如果一开始就把 region 引入 proposal 轴，窗口会变得过碎，而且同一只手在同一时段内可能会同时接近多个 region，导致 proposal 数量膨胀。

因此当前设计采用：

```text
先 hand-time proposal
后 region attribution
```

这种方式更稳定，也更符合 Stage2 的目标，即先找对“哪只手、哪个时间段值得修”，再判断它主要对应哪个目标区域。

### 3.2 raw segment 的生成

在 hand-level binary mask 上，系统会扫描连续的 contact runs，并提取原始时间段：

```text
[raw_start_frame, raw_end_frame)
```

每个 raw segment 至少包含：

- `dataset_row_index`
- `sample_index`
- `dataset_key`
- `hand_side`
- `hand_side_id`
- `raw_start_frame`
- `raw_end_frame`
- `raw_length`
- `center_frame`

因此，raw segment 的本质是：

```text
一个 hand-specific 的 coarse contact time segment
```

### 3.3 gap merge

为了避免短暂接触中断把一个本应连续的接触事件切得过碎，设计中使用 `gap_merge` 对相邻段进行合并：

- 若后一段的起点与前一段终点的间隔不超过 `gap_merge`
- 则将两段视为同一 contact event

这一步的作用是：

1. 提高时间连续性
2. 降低对短时 mask 抖动的敏感性
3. 使 raw segment 更接近语义上的局部接触事件

### 3.4 最短长度过滤

raw segment 生成后，还会通过 `raw_L_min` 进行过滤：

- 太短的段被视为不稳定或训练价值过低
- 只保留长度不小于阈值的 raw segments

这一步的作用是进一步去除碎片化短段，使后续 window 更稳定。

因此，从 coarse motion 到 raw segment 的核心逻辑可以概括为：

```text
从 coarse hand-level contact 序列中，
提取连续、稳定、具有最小时间长度的 hand-time contact events。
```

## 4. 从 Raw Segment 到 Region Attribution

### 4.1 为什么 region attribution 放在 segment 之后

当前实现中，region 不是 proposal 轴，而是 attribution 结果。  
这意味着系统先确定：

```text
哪只手
在哪段时间
```

值得修正，然后再判断：

```text
这段时间里主要接触的是哪个 target region
```

这样做的优点是：

1. proposal 更稳定，不会因为 region 划分过细而爆炸。
2. 同一 hand-time segment 可以自然保留 top-k regions，而不是被硬拆成多个 proposal。
3. 更适合后续 refiner 用 primary region + top-k regions 做条件输入。

### 4.2 attribution 的计算方式

对于每个 raw segment，系统会在该时间段内，对 actor 的每个 target region 统计：

- `num_contact_frames`
- `mean_min_dist`
- `min_dist`

然后按以下优先级排序：

```text
num_contact_frames desc
-> mean_min_dist asc
-> min_dist asc
-> region_id asc
```

### 4.3 attribution 的输出

排序后，segment 会得到以下 region 相关字段：

- `primary_target_region`
- `primary_target_region_id`
- `secondary_target_region`
- `secondary_target_region_id`
- `topk_target_regions`
- `topk_target_region_ids`
- `topk_region_scores`
- `region_score_table`

因此，region attribution 的本质是：

```text
给每个 hand-time segment 分配一个主 region，
同时保留若干备选 top-k regions，
从而把严格单标签 region assignment 放宽成可解释的 region ranking。
```

### 4.4 作用

这一步的作用非常重要：

1. 它为 strict primary-region 分析提供主标签。
2. 它为 top-k attribution 提供更鲁棒的 region 表达。
3. 它为后续 refiner 提供 `primary + top-k` 的条件信息。

也就是说，当前 selector 的强项不是“一步选对唯一 region”，而是：

```text
先稳定覆盖 hand/time，
再用 top-k region attribution 表达局部目标区域。
```

## 5. 从 Attributed Segment 到 Fixed Window

### 5.1 为什么需要 fixed window

raw segment 的长度是可变的，但当前 refiner 需要统一长度的局部输入单元。  
因此，selector 不直接把 raw segment 送入 refiner，而是进一步转成 fixed-size windows。

这一步的目的有三点：

1. 统一训练与推理输入形状
2. 保持 local refinement 的固定时间尺度
3. 让不同 contact events 都能映射到统一窗口表示

### 5.2 window 的生成方式

对于一个 attributed segment，系统会先确定中心点，再根据 `window_size` 扩成固定窗口。

普通情况下：

- 用 `center_frame` 作为中心
- 在有效序列范围内向两侧展开为固定长度窗口

若序列本身长度小于窗口长度，则窗口退化为整段有效序列。

### 5.3 长 segment 的处理

对于特别长的 raw segment，当前实现不是只生成一个窗口，而是生成两个窗口。  
具体做法是：

- 若 `raw_length > 45`
- 则在该 segment 的大约 `1/3` 和 `2/3` 位置各取一个中心点

这样做的目的是：

1. 避免超长接触事件只被一个窗口粗糙覆盖
2. 提高长时间持续接触的局部覆盖能力
3. 仍保持 fixed-window 的统一形式

### 5.4 fixed window 的字段

每个 window 会保留：

- 原始 segment 信息
- `start_frame`
- `end_frame`
- `center_frame`
- `model_window_size`
- 已有的 primary / secondary / top-k region attribution 信息

因此，window 不是一个全新对象，而是：

```text
一个带有 hand-time-region attribution 的固定长度局部裁剪单元
```

## 6. Window Caps、排序与最终输出

### 6.1 为什么需要 caps

即使经过前面的过滤，一个序列中仍可能产生多个候选窗口。  
如果不做限制，Stage2 很容易从“局部精修器”退化成“全序列多窗口重写器”。

因此当前设计通过 window caps 强制保持：

```text
small-number, high-value, local refinement
```

### 6.2 排序依据

在截断之前，所有候选窗口会先被排序。  
当前排序优先级大致为：

```text
raw_length desc
-> hand_contact_frame_ratio desc
-> raw_start_frame asc
-> hand_side_id asc
-> target_region_id asc
-> start_frame asc
```

这表示系统更倾向于保留：

- 持续时间更长的 contact event
- hand-level contact 覆盖更高的窗口
- 时间上更靠前、排序更稳定的窗口

### 6.3 两级 caps

当前设计使用两级限制：

- `per_hand_max_windows`
- `per_seq_max_windows`

也就是说：

1. 每只手最多保留固定数量的窗口
2. 每个序列最多保留固定数量的窗口

这一步的作用是：

1. 防止某只手的高频 contact 主导整段序列
2. 防止 contact-rich 样本产生过多窗口
3. 把 Stage2 明确限制在少量局部修正上

### 6.4 最终输出

selector / window 模块最终输出的是一个完整 artifact，而不只是 window 列表。  
其中主要包括：

- `pred_contact_mask`
- `pred_min_region_dist`
- `hand_contact_mask`
- `hand_min_dist`
- `raw_segments`
- `windows`
- `selector_sequence_stats`
- `selector_params_json`
- `selector_stats_json`

因此，这个模块的最终结果不仅支持后续训练，也支持：

- subset 统计
- selector audit
- dataset 构建
- 可视化和诊断

### 6.5 总结

当前 selector / window 的设计可以概括为：

```text
先从 coarse motion 中重建 shape-aware coarse contact，
再抽取稳定的 hand-time raw segment，
随后做 region attribution，
再变成 fixed-size windows，
最后通过排序与 caps 保留少量高价值窗口。
```

这套设计的本质不是学习 proposal，而是用一个稳定、可审计、局部化的窗口设计算法，为 Stage2 refiner 提供统一而高价值的 local contact refinement 单元。

### 6.6 凝练总结一段话
当前 Stage2 的 window 设计是一个确定性的局部窗口采样器，用于从 Stage1 的 coarse reactor motion 中提取最值得进行 hand-contact refinement 的局部时段，并将其转化为统一的窗口级输入单元；其整体流程为：首先在 restored pair space 中基于 reactor 左右手到 actor 六个目标区域的最小顶点距离构建 coarse contact 表示，用 tau_contact=0.10 将距离二值化为 pred_contact_mask，再在 region 维度做聚合得到 hand-level contact 序列；随后对每个 sample x hand 的二值接触序列提取连续 raw segments，使用 gap_merge=4 合并相邻短间隔片段，并通过 raw_L_min=2 过滤过短片段；然后在每个 raw segment 内对六个 actor target regions 统计 num_contact_frames、mean_min_dist 和 min_dist，按 num_contact_frames 降序、mean_min_dist 升序、min_dist 升序进行排序，得到 primary region、secondary region 和 top-k regions，其中 top_k_regions=3；之后将带 region attribution 的 raw segment 转换为 fixed windows，默认 window_size=30，普通 segment 以中心帧开一个窗口，若 raw_length > 45 则在约 1/3 和 2/3 位置各开一个窗口；最后对候选窗口按 raw_length 降序、hand_contact_frame_ratio 降序及时间顺序排序，并施加两级窗口上限，即 per_hand_max_windows=2、per_seq_max_windows=3，从而得到最终的局部 window artifact，供后续 refiner 作为统一的窗口级输入使用。