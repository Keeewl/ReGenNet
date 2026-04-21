# Stage2_Refine_v1 Summary

## 1. 一句话定位

`stage2_refine_v1`（当前来源里的实现更准确地说是 **Stage2-lite / joint-based local refinement baseline**）是一个建立在 `reaction_data` 桥接包之上的、**手部中心（hand-centric）**、**关节级（joint-based）**、**局部残差修正（local residual refinement）** 两阶段系统。它的目标不是重新生成整段反应动作，而是在 Stage1 生成的 coarse reactor motion 上，先用确定性窗口选择器找出“可能值得修”的局部时间窗口，再在这些窗口内预测小幅残差，从而改善手部接触相关的局部表现。

它的设计关键词可以概括为：

- `reaction_data` 统一数据桥
- `restored_pair_space` 本地几何空间
- `deterministic hand-centric windows`
- `joint-based local features`
- `lightweight residual refiner`
- `local contact eval + global motion eval`

但同时，它也在代码注释里明确承认：这套实现只是 **baseline / Stage2-lite**，还不是最终的 **region-aware / mesh-aware / strict-contact** 方案。

---

## 2. 整体框架

整体链路可以概括为：

```text
Stage1 frozen model
    ↓
build_reaction_data.py
    ↓
reaction_data.npz
    ↓
ReactionDataDataset / collate
    ↓
DeterministicWindowSelector
    ↓
JointFeatureBuilder
    ↓
JointLocalRefiner
    ↓
refined_pack.npz
    ↓
local_contact.py + global_motion.py + summary.py
```

也就是：

1. **Stage1 冻结采样**一次，生成 actor/coarse/gt 等字段并打包为 `reaction_data`。
2. Stage2 训练/推理时直接读取 `reaction_data`，不再依赖旧的 `coarse_cache / restored_cache / blueprint_cache` 命名体系。
3. 在 batch 内先做 **窗口选择**，把一整段序列压成若干局部窗口。
4. 对每个窗口构造 hand-centric 的局部 joint 特征。
5. 用轻量 Transformer 风格网络预测局部残差。
6. 把窗口级 refined 结果融合回完整序列，导出 `reactor_refined`。
7. 用两套评估协议检查：
   - restored-space 下的 **local/contact** 指标
   - inverse-restore 回 Stage1 processed space 后的 **global/STGCN** 指标

---

## 3. 数据入口与空间定义

### 3.1 `reaction_data`：Stage1 → Stage2 的新桥

当前实现用 `reaction_data` 作为新的主数据入口，取代旧的 cache 命名方式。它由 `build_reaction_data.py` 生成，核心字段包括：

- `actor_motion`
- `reactor_gt`
- `reactor_coarse`
- `lengths`
- `sample_indices`
- 以及一组 restored-space metadata

这一设计的目的，是让新的 `refine/` 管线自洽、独立，同时还能复用 Stage1 的采样链。`README.md` 明确把这条构建链标记为当前 “known good” 的 Stage2-lite data-entry result。

### 3.2 `restored_pair_space`：局部几何操作的主空间

`reaction_data` 的 schema 把默认空间定义为 `restored_pair_space`。这意味着本地 contact / window / feature 这条链，默认是在恢复了 pair-level 平移、ground offset、shape/gender metadata 的空间中做的，而不是直接在 Stage1 处理空间里做。

### 3.3 本地与全局协议分离

这套实现里非常重要的一点是：

- **local/contact** 评估在 `restored_pair_space`
- **global/STGCN** 评估在 `stage1_aligned_processed_space`

后者通过 `global_motion.py` 先做 inverse restore，把 transl 与 ground offset 减回去，再送入 STGCN。代码和 summary 都明确强调：**不要把 restored-space 的 local metrics 和 processed-space 的 global metrics 直接混在一起解释。**

---

## 4. 核心定义

## 4.1 任务定义：不是重生成，而是局部 refine

`network.py` 明确写了，当前网络的目标不是重新生成完整序列，而是预测 **low-amplitude local residual**。因此 Stage2_v1 的基本哲学是：

- coarse motion 是主输入
- Stage2 只改一小部分局部片段
- 输出是 `delta_local`
- 再与 coarse 局部片段相加得到 `refined_local`

这使它天然偏向 **conservative refinement**：更像局部修补，而不是接触过程重建。

## 4.2 作用域定义：hand-centric

整个 v1 的作用域是明显 hand-centric 的：

- selector 只围绕左右手与 actor 各个 target parts 的几何关系来做窗口选择
- joints.py 中的局部关节作用域只包括：
  - wrist + hand joints
  - 少量 elbow / shoulder / upper torso support joints
- 明确 **不包含 transl/root slot**

也就是说，v1 的局部修正对象不是“接触相关全身协同”，而是“以手为中心、带少量支撑关节”的局部补丁。

## 4.3 target 定义：左右手 × 6 个预定义 actor parts

旧 selector 与 feature builder 都依赖一个固定的 target-part 体系：

- `torso_head`
- `lower_body`
- `left_arm`
- `right_arm`
- `left_hand`
- `right_hand`

这意味着 v1 的接触语义不是 mesh/contact-region，而是 **预定义 joint-part semantic buckets**。

## 4.4 状态定义：`strict` / `near`

旧版 selector 的时间 ROI 定义不是简单的 binary contact，而是：

- 先对每帧、每只手、每个 target part 计算多个 joint-level 几何与运动 cue
- 聚合为 `selection_score`
- 再构造 `strict_scores` 和 `near_scores`
- 用 strict 生成 anchor，再向前后用 near 扩展 segment

因此它抓到的首先是“几何/接触关键事件”，而不是严格定义的 GT contact segment。

---

## 5. 关键实现模块

## 5.1 `build_reaction_data.py`

这个脚本负责：

- 加载冻结的 Stage1 checkpoint
- 用 Stage1 模型对数据集做一次采样
- 保存 actor、coarse、gt、lengths、sample index
- 如果 restoration metadata 可用，则直接把 actor/gt/coarse 都 restore 到 restored pair space
- 同时把 dataset_key、frame index、betas、gender、body model type 等 metadata 一起打包

其意义是：让 Stage2 不需要再依赖旧版多套 cache/blueprint 文件，而只认一份 `reaction_data` 包。

## 5.2 `DeterministicWindowSelector`：确定性 hand-centric 选窗

这是 v1 最核心、也最有代表性的模块之一。

### 它做了什么

1. 先把 actor/coarse motion restore 到 restored pair space。
2. 用 `Rotation2xyz_x` 把 motion 转到 xyz joints。
3. 对每只手、每个 target part，计算：
   - target distance
   - other-part distance margin
   - contact ratio
   - relative speed
   - target motion
   - approaching score
4. 聚合得到 `selection_score`。
5. 对 target id 做 smoothing。
6. 用 `strict_scores` 提取 anchor，用 `near_scores` 向前后扩展。
7. 形成 variable-length `RawSegment`。
8. 再转成 fixed-length model windows。
9. 最后按每手/每序列上限做裁剪。

### 默认超参特征

默认 `WindowConfig` 里，比较关键的参数有：

- `strict_score_threshold = 0.62`
- `near_score_threshold_pre = 0.42`
- `near_score_threshold_post = 0.34`
- `raw_L_min = 6`
- `raw_L_max = 24`
- `model_W = 16`
- `gap_merge = 2`
- `per_hand_max_windows = 3`
- `per_seq_max_windows = 6`
- `strict_contact_distance = 0.08`
- `near_contact_distance = 0.18`

这说明 v1 的设计目标更像是：**用紧凑、小窗口、可审计的 deterministic 机制，抓出疑似接触关键片段。**

### 它的问题在哪里

虽然它“简单、稳定、可审计”，但代码自己也明说：它只是 **baseline joint-based window selector**，还不是 region-aware/mesh-aware selector。它的问题也正来自此：

- 依赖 hand-to-part 的固定语义桶
- 依赖 joint-level proxy
- 依赖 strict/near 这套人为状态
- 目标是“contact-critical event”，不是 strict GT contact segment

这正是后来被认为“更像 hand-event selector，而不是 contact selector”的核心原因。

## 5.3 `JointFeatureBuilder`：joint-based 局部特征构造

这个模块把选出来的 windows 变成网络可吃的张量。

### 核心做法

- 仍然要求 restoration metadata 与 restored pair space
- 把 actor/coarse/gt 转成 xyz
- 按 hand side 取局部 joint scope：
  - wrist + hand joints + 少量 support joints
- 按 target part 取 target joints
- crop 出固定长度局部 motion / xyz 窗口
- 把 target joints pad 到统一长度 `MAX_TARGET_JOINTS`
- 构造：
  - `coarse_local`
  - `gt_local`
  - `actor_target_local`
  - `actor_target_mask`
  - `source_joint_ids`
  - `time_mask`
  - `target_summary_feat`

### `target_summary_feat`

这是一个非常典型的“轻量关系摘要”实现。它不是 mesh token，也不是 proposal stack，而是把以下信息按帧打包：

- hand center 与 target center 的相对位移
- 距离
- 相对速度
- 最小 joint distance
- 再拼接 `window_state` one-hot 与 `target_part` one-hot

这说明 v1 的 feature 设计已经在尝试“关系感知”，但仍然停留在 **局部 joint crop + 轻量摘要**，不是 mesh-aware 表征。

## 5.4 `JointLocalRefiner`：局部残差网络

网络结构本身是一个轻量级 Transformer 风格局部 refiner。

### 输入流

- `coarse_local`：主流
- `actor_target_local`：条件流
- `target_summary_feat`：关系调制摘要

### 模块结构

每个 `RefinerBlock` 包括：

- Temporal self-attention
- Cross-attention（reactor 对 actor target）
- Spatial self-attention
- Relation FiLM
- FFN

### 输出

- 经过 `delta_head` 得到 `raw_delta`
- 再通过 `tanh * delta_scale` 得到低幅 `delta_local`
- 与 coarse 相加得到 `refined_local`

这套网络非常清楚地说明：

- 它是 **轻量局部修正器**
- 不是 diffusion refiner
- 不是 mesh-aware refiner
- 不是全身协同 refiner

因此从设计上讲，它天然更适合“小修补”，而不是“显著改善接触物理性”。

## 5.5 `JointRefinementLoss`：保守型 joint-based 损失

损失函数由 4 项组成：

- `L_res`：局部残差重建
- `L_smooth`：时间平滑
- `L_contact_proxy`：只惩罚 refined 比 coarse 更差的 contact proxy 距离
- `L_identity`：抑制无谓改动

即：

```text
L = lambda_res * L_res
  + lambda_smooth * L_smooth
  + lambda_contact * L_contact_proxy
  + lambda_id * L_identity
```

这里最关键的结构性特征是：

1. 它是 **joint-based**，不是 mesh loss。
2. contact 项不是“主动把接触拉好”，而是“防止 refined 比 coarse 更差”。
3. identity regularization 明显偏保守。

这解释了为什么 v1 更像“局部保守微调器”，而不是“接触驱动优化器”。

---

## 6. 训练、推理与评估

## 6.1 训练：`Stage2LiteTrainer`

训练流程是：

1. 读 `reaction_data`
2. batch 内先跑 `DeterministicWindowSelector`
3. 如果没有任何窗口，这个 batch 直接记为 empty-window batch
4. 否则用 `JointFeatureBuilder` 构造窗口特征
5. 跑 `JointLocalRefiner`
6. 用 `JointRefinementLoss` 回传
7. 定期保存 checkpoint

这说明窗口选择是训练时在线完成的，不是预先静态缓存好的“最终 label”。

## 6.2 推理：`Stage2LiteInferRunner`

推理器提供了一套很完整的 subset / manifest / coverage 机制。

它支持：

- `fixed`
- `random`
- `stratified`

三种采样模式，并会保存：

- `subset_manifest.json`
- `coverage_report.json`
- 可选 `debug_stats.json`
- `refined_pack.npz`

也就是说，v1 在实验工程化上其实做得比较完整：它已经能较规范地控制 infer subset、覆盖率统计和复现实验协议。

## 6.3 本地接触评估：`local_contact.py`

这是 v1 里“主 Stage2 指标”的实现位置，但它自己在文件头就说得很清楚：

- 这是 **restored-space 下的主 contact-oriented evaluation**
- 但它实现的是 **lightweight joint-based contact proxy**
- 并没有使用 mesh-aware contact 定义

具体来说，它是：

- 取 reactor 两只手的 joints
- 取 actor 全身 joints
- 算最小 pairwise joint distance
- 再从中得到：
  - `hand_cd`
  - `contact_ratio`
  - `avg_contact_duration`
  - `contact_frequency`
  - `region_hand_dist`
  - `penetration_rate`
  - `penetration_depth`

这套指标已经开始朝“Stage2 更看 contact/physicality”靠拢，但 GT 定义和 metric carrier 仍然是 joint proxy。

## 6.4 全局动作评估：`global_motion.py`

`global_motion.py` 的定位非常明确：

- 不是主 Stage2 指标
- 而是 auxiliary check
- 用 inverse restore 回到 Stage1 processed space
- 再跑 STGCN 计算：
  - accuracy
  - diversity
  - multimodality
  - fid

它的主要作用是检查：**局部 refiner 有没有把整体动作分布搞坏。**

## 6.5 汇总：`summary.py`

summary 模块把：

- local json
- global json
- infer manifest
- coverage report

汇总成一个 summary，并明确写出三条原则：

1. local/contact 是主 Stage2 hand/contact evaluation
2. global/STGCN 是 auxiliary recognition/distribution check
3. 不要把两套协议直接混在一起解释

这说明 v1 在“指标解释框架”上其实已经形成了比较成熟的工程习惯。

---

## 7. v1 的核心优点

从工程与研究探索角度，v1 有几个明显优点：

### 7.1 链路完整

它不是一堆零散模块，而是一条完整的 Stage2-lite 管线：

- 数据桥
- 训练
- 推理
- subset protocol
- local/global eval
- summary

### 7.2 与 Stage1 解耦但兼容

`reaction_data` 把 Stage1→Stage2 的边界定义清楚了，这对后续继续改 Stage2 很重要。

### 7.3 确定性、可审计

窗口选择虽然简化，但 deterministic，便于 audit 和 debug。

### 7.4 轻量

局部窗口 + 小幅残差网络的设计，训练和推理成本都可控。

### 7.5 已经意识到“物理/接触”和“全局分布”应分开评估

这为后续演化到 v2 提供了正确的评价框架基础。

---

## 8. v1 的失败经验与结构性问题

这部分是最重要的。

### 8.1 任务定义偏了：更像 hand-event refinement，不是 strict contact refinement

selector 的目标并不是 strict GT contact segment，而是 **基于手与 target part 的 joint-geometry/motion cue 的 contact-critical event**。这使它容易命中“事件感强”的片段，而不一定是真接触片段。

### 8.2 selector 与 audit 共享同类 proxy，形成“自洽但偏题”

这是 v1 最大的问题之一。

`window_audit.py` 明确写了：当有 `gt_motion` 时，它是“再对 GT 跑一遍同一个 selector”，因此只是 **selector-vs-selector proxy audit**，而不是 strict GT contact-label evaluation。

也就是说：

- selector 用 proxy 定义目标
- audit 也用同类 proxy 验证 selector

整个系统会出现一种危险的“内部自洽”：看起来窗口指标不错，但并不一定真的抓到了你想要的 contact process。

### 8.3 作用域太窄：只修手及少量 support joints

局部 joint scope 明确不含 transl/root，只覆盖 hand + elbow/shoulder/upper torso 的少量 support joints。

这意味着：

- 如果接触问题本质上需要 body support、重心调整、transl 配合
- v1 从定义上就修不到

也就是说，v1 很可能把“全身协同接触问题”错误地简化成“手部局部补丁问题”。

### 8.4 固定 target-part 过早绑定，容易放大误差

整个链路是围绕 `hand -> target_part` 的固定语义桶构建的：

- selector 选 target part
- feature builder crop target joints
- network 以 target part 为条件
- summary feature 里还有 target-part one-hot

一旦 target 早期判断有误，后面整条链路都会偏。

### 8.5 feature / loss / eval 都还是 joint-based proxy

v1 虽然已经开始讲“contact-oriented”，但本质上：

- feature 是 joint crop + joint relation summary
- loss 的 contact 项是 joint-distance proxy
- eval 的 local contact 仍是 hand joints 到 actor joints 的最小距离

所以系统虽围绕“接触”组织，但其 **carrier 仍然是 joints**，不是 mesh/contact region。

### 8.6 loss 太保守，偏防退化，不偏主动改善

`L_contact_proxy` 惩罚的是：refined 不要比 coarse 更差；
`L_identity` 进一步抑制改动幅度。

这在防止训练崩坏方面是有益的，但也意味着：

- 模型更倾向于不改或少改
- 对明显改善 contact 的推动力偏弱

因此即使 Stage2 不严重破坏动作分布，也未必会显著改善 contact/physicality。

### 8.7 window 太短，片段语义可能不完整

默认 `model_W = 16` 帧，明显偏短。

对于握手、拥抱、跟手等持续型接触过程，这种长度更容易只截到“接触瞬间”或“局部几帧”，而不是完整的接触语义片段。后续讨论里你们也逐步意识到：像 handshake 这种语义，接触主段可能接近几十帧，16 帧窗口很可能不足以支持显著 refine。

---

## 9. 为什么 v1 最终被认为需要升级到 v2

综合来看，v1 的问题不是“某个模块还不够强”，而是：

**整条链都建立在一个 hand-centric / joint-proxy / event-driven 的 formulation 上。**

也就是说：

- 数据桥与评估分层是对的
- 但 selector 定义偏了
- target-part 假设偏硬
- feature / loss / eval 都被 joint proxy 绑定
- 作用域又太窄

因此后续往 `refine_v2` 走，真正需要升级的不是某个单点，而是按下面顺序整体升级：

```text
selector → window stats → contact labels / subset → feature → network → loss → eval
```

而不是继续在旧的 `strict/near + joint proxy + hand-centric` 定义上小修小改。

---

## 10. 最终总结

`stage2_refine_v1` 的历史作用，不是提供了一个已经成熟的接触优化框架，而是完成了三件非常重要的事情：

1. **把 Stage1→Stage2 的数据桥和工程协议搭起来了。**
2. **验证了“局部窗口 + 轻量残差修正”这条 Stage2 形态是可运行的。**
3. **通过失败经验暴露出：仅靠 hand-centric joint-proxy event formulation，难以真正实现你们想要的 contact-oriented / mesh-aware / physical refinement。**

所以 v1 更准确的历史定位应该是：

> 一个工程上完整、研究上有启发性、但在任务定义与表示层面仍然偏 proxy 的 Stage2-lite 基线。

它最重要的价值，不只是它“做成了什么”，更在于它明确告诉了后续 v2：

- contact 的 GT 定义必须更严格
- selector 不能继续只抓几何事件
- carrier 需要从 joints 升级到 mesh/region
- 评估必须从 proxy audit 升级到 strict contact-label audit
- 局部 refine 的上下文窗口也必须重新设计
