# Stage2 Refine V2 总述

## 1. 这个 summary 目录的结构是否完整、合理

目前目录为：

- `01_definition.md`
- `02_selector_window.md`
- `03_feature.md`
- `04_model.md`
- `05_loss.md`
- `06_evaluation.md`
- `stage2_refine.md`

这套顺序整体上是合理的，符合 Stage2 从任务定义到实现细节的阅读路径：

```text
任务定义
-> 选择哪些局部窗口需要修
-> 构造给 refiner 的条件特征
-> refiner 模型本体
-> 训练目标
-> 评价方式
```

从“帮助理解代码”和“后续写论文”两个目标看，这个结构有三个明显优点：

1. 它按实现链路展开，而不是按文件夹机械罗列，便于建立整体因果关系。
2. 它把最核心的 Stage2 问题拆成 selector / feature / model / loss / eval 五层，适合后续写 method。
3. 它天然支持把论文中的 `method`、`training objective`、`evaluation` 三大部分直接映射出来。

如果 `01_definition.md` 中已经吸收以下内容：

- `Stage1 frozen model`
- `reaction_data` 作为上游定义
- `hand-contact-rich subset`
- `restore-shape / restored pair space`
- `actor / reactor region definition`

那么这套方法结构对于论文 method 叙事来说，已经是完整的。

在这种组织方式下，当前 summary 的主干变成：

```text
definition
-> selector/window
-> feature
-> model
-> loss
-> evaluation
```

这个主干是合理且足够闭合的。

仍然建议补的一层，主要不是方法定义缺失，而是：

### 1.1 实验结论与版本演化总结

现在 `06_evaluation.md` 更适合写“如何评估”。  
但从论文准备和后续答辩角度，还需要一份“当前阶段结论”：

- 为什么 `exp5` 是当前最好 baseline
- 为什么 selector 不是当前瓶颈
- 为什么 geometry v2 / proxy contact loss 没有明显超过 `exp5`
- 当前 bottleneck 已经转移到 interaction modeling 和 task alignment

这部分最好单独再有一篇：

- `07_experiment_findings.md`
- 或者 `07_ablation_and_conclusion.md`

### 1.2 当前结构的总体判断

因此我对当前结构的结论是：

- `合理`: 是
- `高效`: 是，阅读路径清楚
- `完整`: 对方法定义来说，已经基本完整

更准确地说：

```text
这是一套已经能够支撑 Stage2 method 写作的结构；
如果后面再补一篇实验结论总结文档，会更利于论文结果分析与答辩表达。
```

## 2. Stage2 refine_v2 的核心定义

`refine_v2` 不是第二个完整动作生成器，而是一个建立在 Stage1 输出之上的：

```text
局部窗口级
hand-centric
contact-oriented
residual refinement
```

系统。

它的基本任务不是“重新生成整段 reactor motion”，而是：

```text
给定 Stage1 生成的 coarse reactor motion，
在疑似接触相关的局部时间窗口内，
预测小幅残差修正，
让 reactor 的 hand/arm 与 actor 的目标区域接触更接近 GT。
```

这也是为什么当前代码和设计文档都不断强调：

- Stage1 负责提供可用的 coarse motion
- Stage2 负责修正局部 contact-relevant errors

当前高层诊断也明确认为，Stage2 的正确角色应该是：

```text
window-level
local
hand-centric
contact-oriented
residual
```

而不是 full-sequence regeneration。

## 3. Stage2 的输入、输出与工作空间

### 3.1 输入来源

Stage2 的上游输入不是原始 dataset，而是 Stage1 构建出的 `reaction_data` 包。  
其中最核心的三个 motion 字段是：

- `actor_motion`
- `reactor_coarse`
- `reactor_gt`

也就是说，Stage2 学的是：

```text
coarse -> refined
```

不是：

```text
text / action -> motion
```

### 3.2 工作空间

Stage2 的 contact、window、训练样本都要求在：

```text
restored_pair_space
```

里对齐。

这点非常关键，因为 Stage2 关心的是：

```text
reactor 手部
相对于
actor 身体目标区域
```

的局部几何关系。  
如果不在统一恢复后的双人空间里，最小距离、接触时长、接触区域都会失真。

### 3.3 输出定义

模型最终输出不是完整新序列，而是窗口级残差：

```text
pred_delta_motion_window
pred_motion_window = coarse_motion_window + pred_delta_motion_window
```

在推理和可视化阶段，再把这些局部 refined windows stitch 回整段序列。

## 4. Stage2 的完整闭环

从代码实现看，`refine_v2` 现在已经形成了一个相对闭合的 Stage2 流程：

```text
Stage1 frozen model
-> reaction_data.npz
-> GT binary contact labels
-> deterministic selector windows
-> selector audit
-> action-type statistics
-> contact-rich subset
-> rerun selector on subset
-> window dataset
-> local residual refiner
-> contact-oriented evaluation
-> visualization / diagnosis
```

这个闭环的意义在于：

1. Stage2 已经不再依赖旧 `stage2_old` 运行时主线。
2. 训练、评估、可视化、审计都围绕同一套 `reaction_data + restored_pair_space` 定义展开。
3. 当前已经可以把 Stage2 作为一个独立模块来分析，而不是散乱实验脚本的集合。

## 5. 各模块的核心职责

### 5.1 GT contact label

GT label 的定义方式是：

1. 将 actor / reactor motion 转成 SMPL-X vertices
2. 对 reactor 左右手到 actor 六个 body regions 计算逐帧最小距离
3. 若距离小于 `tau_contact`，则标记为 contact

因此 GT supervision 不是抽象的动作标签，而是：

```text
hand x target-region x frame
```

的局部几何接触真值。

### 5.2 Selector window

selector 不是学习式 proposal，而是确定性规则模块。  
它从 coarse contact 中提取 hand-time segments，做 gap merge 和短段过滤，再扩成固定窗口，并为每个窗口分配：

- selected hand
- primary target region
- top-k target regions

这个设计说明当前 Stage2 的策略是：

```text
先冻结“哪里值得修”
再集中学习“怎么修”
```

而不是同时学习 proposal 和 refinement。

### 5.3 Contact-rich subset

Stage2 训练也不是在全量序列上进行，而是先按 action type 统计 contact richness，再建立 sequence-level subset，并重点保留：

```text
GT+ / Pred+
```

主桶。

这一步非常重要，因为它本质上定义了 Stage2 的训练域：

```text
高接触密度、selector 可覆盖、对 refinement 真正有价值的样本
```

而不是整个 Inter-X train。

### 5.4 Window dataset

当前 refiner dataset 的最小样本单位是：

```text
one hand-time selector window per sample
```

每个样本保留：

- actor / coarse / gt motion crop
- coarse region contact mask / min distance
- GT region contact mask / min distance
- hand / primary region / top-k region metadata
- 可选 geometry feature cache

这一定义直接把 Stage2 限定成了“局部接触窗口修正器”。

### 5.5 Refiner model

当前模型本体是一个：

```text
window-level temporal transformer residual refiner
```

它做的事情可以概括成：

1. 将 `coarse_motion_window` 和 `actor_motion_window` 编码成时间 token
2. 将 hand / region / top-k / coarse contact / optional geometry 编码成 condition
3. 用 temporal attention + actor cross attention + condition modulation 融合上下文
4. 输出局部 residual delta

本质上它不是 spatial transformer，也不是 diffusion。  
它是一个带条件调制的局部时间序列残差修正器。

### 5.6 Scope control

当前 Stage2 一个很成功的设计点是 scope control。  
无论是 residual scaling 还是 loss weighting，都在强制模型遵循下面的修正优先级：

```text
hand / same-side arm 变化最大
other upper 次之
torso / root / translation / lower body 尽量保守
```

这符合 Stage2 的角色：  
它应该改善接触，而不应该变成 full-body rewrite 模块。

## 6. 当前训练目标的本质

虽然 Stage2 的最终目标是改善 contact，但当前训练目标仍然主要是：

```text
让 pred motion 更接近 GT motion
```

只是通过以下机制把重点推向接触相关区域：

- group-weighted motion loss
- contact-frame weighted loss
- hand/arm contact-weighted loss
- smooth residual regularization
- boundary translation anchor
- 可选 contact geometry regularization
- 可选 GT-relative overclose regularization

所以当前 Stage2 的训练逻辑更准确地说是：

```text
以 motion reconstruction 为主干，
以 contact relevance 为权重偏置，
而不是完全直接优化真实 contact geometry。
```

这也正是当前阶段的主要瓶颈之一。

## 7. 当前评估定义

Stage2 的 eval 比 train 更直接面向任务目标。  
评估时会重新根据 refined motion 计算 hand-to-region 距离，然后比较：

- coarse vs GT
- refined vs GT

核心指标包括：

- `all_valid_dist_l1_improvement`
- `gt_contact_contact_dist_improvement`
- `refined_contact_f1`
- `topk_refined_contact_f1`
- contact duration / frequency / jitter error
- `surrogate_penetration_*`

因此，Stage2 的真实性能判断并不只看 motion error，而主要看：

```text
refined 是否比 coarse 更接近 GT contact state
```

## 8. 当前阶段的核心结论

从现有实现和实验总结来看，当前可以下一个比较稳的结论：

### 8.1 Stage2 整体框架已经成立

目前已经证明以下部分不是主要瓶颈：

- GT contact labels
- restored-space processing
- selector/window
- subset construction
- train/eval/visualization pipeline

也就是说：

```text
Stage2 的整体框架是有效的
```

### 8.2 当前最好基线是 exp5

当前最稳的 practical baseline 是：

```text
refiner_v2_exp5_scope_geom_10k
```

它说明：

- coarse motion 可以被稳定改善
- contact distance 可以进一步接近 GT
- contact F1 可以稳定高于 coarse
- Stage2 的 scope-aware residual design 是有效的

### 8.3 当前主要瓶颈已经转移到模型本身

现在真正限制上限的，不再是 selector 或数据接口，而是：

```text
模型如何表达 hand-target interaction
模型如何更直接对齐 contact geometry
模型如何真正用好 geometry features
```

换句话说，当前 Stage2 的难点已经从：

```text
工程打通
```

转移到：

```text
representation / interaction modeling / task alignment
```

## 9. 对论文写作的建议定位

如果后面要写论文，当前 Stage2 最适合被表述为：

```text
一个轻量、可插拔的局部 contact refinement stage
```

它的核心卖点不是“生成能力”，而是：

1. 站在 Stage1 之上工作的 modular refinement design
2. 面向 hand-target contact 的局部窗口建模
3. 通过 restored pair space 建立几何一致的接触定义
4. 通过 deterministic selector 将计算集中在真正值得修的局部窗口
5. 通过 scope-aware residual control 在改善 contact 的同时避免全身动作被破坏

## 10. 后续 summary 文档建议分工

基于当前结构，后续每个文件建议写成：

- `01_definition.md`
  Stage2 的任务定义、输入输出定义、restored pair space 定义、局部窗口定义
- `02_selector_window.md`
  GT contact label、selector 规则、top-k attribution、subset rerun、selector audit
- `03_feature.md`
  refiner 输入样本、geometry cache、condition fields、top-k score encoding
- `04_model.md`
  condition encoder、temporal transformer、actor cross attention、residual heads / boost / group gating
- `05_loss.md`
  motion loss、contact-weighted loss、smooth、boundary、geometry regularization、overclose regularization
- `06_evaluation.md`
  window eval、contact eval、penetration surrogate、breakdown metrics

另外建议后面补两个：

- `02_data_subset.md` 或将 `02_selector_window.md` 扩成 `02_data_and_selector.md`
- `07_experiment_findings.md`

## 11. 当前总评

一句话总结当前的 Stage2：

```text
refine_v2 已经从“实验脚手架”进化成了一个定义清晰的 Stage2 框架：
它以 Stage1 coarse reactor motion 为输入，
以 restored-space hand-target contact 改善为目标，
通过 selector 选出局部窗口，
再用 scope-aware residual refiner 做轻量修正，
当前框架已经可用，主要瓶颈已不在数据与流程，而在模型对 contact interaction 的表达能力。
```
