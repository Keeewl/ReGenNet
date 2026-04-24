# Stage2 Refine V2 定义

## 1. Stage2 的整体定义、目标与框架

`refine_v2` 对应的 Stage2 不是一个新的整段动作生成器，而是一个建立在 Stage1 输出之上的局部精修阶段。  
它的基本角色可以概括为：

```text
window-level
hand-centric
contact-oriented
residual refinement stage
```

Stage2 的输入不是原始数据集样本，而是由 **冻结的 Stage1 模型** 预先导出的 `reaction_data`。  
其中最核心的输入字段包括：

- `actor_motion`
- `reactor_coarse`
- `reactor_gt`

因此，Stage2 的问题定义不是：

```text
text / action -> motion
```

而是：

```text
given actor motion and Stage1 coarse reactor motion
-> predict refined reactor motion
```

这一定义说明 Stage1 和 Stage2 的职责分工是清晰分离的：

- Stage1：负责生成全局可用的 coarse reactor motion
- Stage2：负责修正 coarse motion 中局部 hand-contact 相关误差

Stage2 的目标也不是重写整段 reactor 动作，而是：

```text
在 Stage1 已生成的 coarse reactor motion 上，
只对与 hand contact 强相关的局部时间窗口做小幅残差修正，
从而让 reactor 与 actor 之间的局部接触更接近 GT 接触。
```

这一定义进一步决定了 Stage2 的三个基本约束：

1. 它是局部窗口级的，而不是 full-sequence 级的。
2. 它是接触导向的，而不是一般动作重建导向的。
3. 它是残差修正的，而不是从零生成的。

从当前实现看，Stage2 已经形成了一个完整闭环：

```text
Stage1 frozen model
-> reaction_data
-> hand-contact-rich subset
-> restore-shape / restored pair space
-> actor/reactor region definition
-> GT contact labels
-> selector windows
-> window feature dataset
-> local residual refiner
-> contact-oriented evaluation
```

这个闭环的核心思想是：

```text
先固定 Stage1 的 coarse generation，
再在一个 hand-contact-rich 的任务子域内，
以 shape-aware、region-aware 的真实接触定义为基础，
对 coarse reactor motion 进行局部接触精修。
```

## 2. Contact-Rich 子集定义

### 2.1 目的

Stage2 的核心任务是改善 hand contact，而不是平均地优化所有交互动作。  
因此训练域应当优先覆盖那些：

- hand-hand contact 明显
- hand-arm / hand-shoulder contact 明显
- hand-body support / pull / help interaction 明显
- face / cheek 等局部高接触动作明显

的样本。

也就是说，Stage2 并不是针对“所有交互动作平均优化”，而是针对：

```text
hand-contact-rich interactions
```

建立一个更聚焦、更适合 contact refinement 的训练与验证子域。

这样做的原因有三点：

1. Stage2 的主要优化目标就是局部 hand contact，训练域应尽量与目标一致。
2. contact-rich 样本能提供更密集、更明确的局部接触监督。
3. 在高接触密度子域中，selector 选出的窗口更有训练价值，false positive 更少。

### 2.2 子集内容

当前 `refine_v2` 使用的是一个从 Inter-X train 中选出的 **15 类 hand-contact-rich action subset**。  
这些动作类别为：

```text
A028 Hand wrestling
A025 Carry on back
A001 Handshake
A009 Sit on leg
A021 Dance
A000 Hug
A008 Pull
A019 Support with hand
A023 Shoulder to shoulder
A035 Help up
A027 Massaging leg
A022 Link arms
A003 Grab
A016 High-five
A034 Kiss on cheek
```

### 2.3 简要分析

这 15 类动作并不是随机挑选的，而是明显偏向局部 hand contact 丰富的 interaction 类型。

其中大致可以分成几类：

- hand-hand direct contact：
  `Handshake`, `High-five`, `Hand wrestling`
- hand-arm / hand-shoulder / upper-body local contact：
  `Grab`, `Link arms`, `Shoulder to shoulder`
- hand-body support / carry / help interaction：
  `Carry on back`, `Support with hand`, `Help up`, `Sit on leg`
- pull / close-range control interaction：
  `Pull`, `Hug`
- local face / cheek contact：
  `Kiss on cheek`
- 具有持续局部接触或多接触段特征的 interaction：
  `Dance`, `Massaging leg`

这个子集的作用不是为了让 Stage2 只适用于少量类别，而是为了把 Stage2 的训练问题明确定义为：

```text
在 hand-contact 密度高、局部几何接触意义强的交互子域中，
学习 coarse -> refined 的局部接触修正。
```

因此，contact-rich subset 是 Stage2 任务定义的一部分，而不是后期附加的数据筛选技巧。

## 3. Restore-Shape 定义

### 3.1 目的

Stage2 的目标是改善真实接触，而不是改善抽象关节轨迹。  
如果只在标准化后的人体表示上比较两人距离，接触几何会丢失真实体型差异，导致：

- 手与目标区域的真实距离不准确
- 接触与非接触的边界不稳定
- 不同体型角色之间的局部接触关系被扭曲

因此，Stage2 明确要求在 **restore-shape / restored pair space** 中定义 contact。

其本质目的就是：

```text
把 Stage2 的 contact refinement 建立在真实双人体型和真实相对空间关系上。
```

### 3.2 怎么恢复

当前实现中的 restore-shape 包含两层含义。

第一层是 **restore pair space**：

- 如果输入已经声明为 `restored_pair_space`，则直接使用
- 否则通过稳定的 restoration metadata 将 actor / reactor motion 恢复到统一双人空间

第二层是 **shape-aware SMPL-X forward**：

- 读取 `actor_betas` / `reactor_betas`
- 读取 `actor_gender_id` / `reactor_gender_id`
- 使用 `body_model_type=smplx`
- 将 motion forward 到对应体型与性别的 SMPL-X vertices

因此，当前 Stage2 并不是在“统一模板人体”上定义接触，而是在：

```text
shape-aware SMPL-X meshes in restored pair space
```

上定义接触。

### 3.3 作用

restore-shape 的作用可以概括为三点：

1. 让 GT contact label 具有真实几何意义。
2. 让 selector window 的 coarse contact 判断建立在真实局部接触距离上。
3. 让 eval 能够直接比较 coarse / refined / GT 的真实 mesh-region 距离变化。

因此，restore-shape 不是附属工程步骤，而是 Stage2 任务成立的几何基础。  
如果没有它，Stage2 的“contact refinement”只能停留在抽象表示层面，而不是物理上更真实的局部接触改善。

## 4. Reactor 和 Actor 的 Region 定义

### 4.1 基本定义

Stage2 并不直接学习“任意全身接触”，而是围绕：

```text
reactor hand
vs
actor target body region
```

来定义局部接触问题。

因此，Stage2 中的 contact 不是无结构的二值事件，而是一个：

```text
hand x target-region x time
```

的问题。

### 4.2 Reactor 的 region 定义

在当前 Stage2 设计中，reactor 侧的接触源只关注：

- `left_hand`
- `right_hand`

也就是说，Stage2 的 GT contact label、selector、feature 和 evaluation 都围绕：

```text
reactor left/right hand
to
actor target regions
```

来定义。

这说明当前 Stage2 是明确的 **hand-centric contact refinement**，而不是泛化的 full-body contact refinement。

### 4.3 Actor 的 6-part region 定义

当前 Stage2 对 actor 侧采用基于 SMPL-X vertex segmentation 的 6-part target region 定义：

- `torso_head`
- `lower_body`
- `left_arm`
- `right_arm`
- `left_hand`
- `right_hand`

默认使用的 segmentation asset 为：

```text
visualize/viewer/part_segm/6_parts/six_parts.pkl
```

其实现方式是：

1. 读取 region map
2. 为每个 target region 提供一个 vertex id 集合
3. 后续所有 hand-to-region 距离都在这些 vertex 集合上计算最小距离

因此，actor 端的接触目标不是“整个人体”，而是被压缩成一个语义明确、可统计、可评估的 6-part region 问题。

### 4.4 作用

这套 region 定义的作用有三点：

1. 它把 contact supervision 从全局 mesh 距离压缩成可解释的 region-level problem。
2. 它让 selector 可以围绕 primary region 与 top-k regions 进行 attribution。
3. 它让 evaluation 可以按 region 做 breakdown，分析哪些 body region 更容易被 refined 改善。

因此，reactor / actor 的 region 定义并不是为了实现方便，而是为了把 Stage2 的局部接触问题结构化。

## 5. Contact 的基本定义

Stage2 中的 GT contact 定义建立在 restore-shape 后的 actor / reactor SMPL-X vertices 上。

对于每一帧，系统计算：

```text
reactor left/right hand
to
actor each target region
```

的最小顶点距离。

若该距离小于阈值 `tau_contact`，则定义为 contact。

因此，GT supervision 的本质是：

```text
a shape-aware, region-aware, frame-wise binary contact definition
```

而不是抽象动作标签，也不是基于关节启发式近似的弱标签。

在这个定义下，Stage2 中最基本的监督对象是：

```text
which hand
contacts which target region
at which frame
```

当前实现同时保留两类量：

- `binary contact mask`
- `minimum hand-to-region distance`

前者决定 contact / non-contact，后者提供更细粒度的局部几何信息。

这个定义的意义在于：

1. 它让 Stage2 的目标从“动作更像 GT”转化为“局部 hand contact 更接近 GT”。
2. 它让 selector、feature、loss、evaluation 都围绕同一个几何接触定义工作。
3. 它把 Stage2 的核心问题明确成一个基于真实体型、真实双人空间、真实局部区域的接触精修问题。
