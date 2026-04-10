# Hand Contact Refinement (HCR)

本目录实现 Stage2 的 Hand Contact Refinement (HCR) 主线，包含手级别接触提案与后续手部精修的基础模块。

## 目录结构

```
model/contact/
  __init__.py
  contact_defs.py
  contact_geometry.py
  proposal_labels.py
  proposal_features.py
  proposal_model.py
  proposal_loss.py
  proposal_events.py
  proposal_windows.py
  refiner_model.py
  refiner_loss.py
  refiner_inputs.py
```

## 模块说明

- `contact_defs.py`: 手部/关节/部位的基础常量与索引定义。
- `contact_geometry.py`: 通用几何工具与 `ContactGeometry`（xyz 转换、topk 距离等）。
- `proposal_labels.py`: `HandContactLabelBuilder`，自动生成帧级标签。
- `proposal_features.py`: `HandContactFeatureBuilder`，构建 hand/part/relation 原始特征。
- `proposal_model.py`: `HandContactProposal`，轻量提案网络。
- `proposal_loss.py`: `HandContactProposalLoss`，分类 + 平滑 + 一致性损失。
- `proposal_events.py`: `ContactEventParser` / `parse_contact_events`，事件段解析。
- `proposal_windows.py`: `ContactWindowBuilder`，事件段到窗口。
- `refiner_model.py`: `HandContactRefiner`，手部精修器。
- `refiner_loss.py`: `HandContactRefinerLoss`，精修损失。
- `refiner_inputs.py`: `ContactWindowSampler`，窗口构造与输入打包。

## 关键张量约定

- 运动输入：`[B, J, 6, T]`
- hand features：`[B, T, 2, Fh_raw]`
- actor part features：`[B, T, 5, Fp_raw]`
- relation features：`[B, T, 2, 5, 8]`
- active logits：`[B, T, 2, 1]`
- target logits：`[B, T, 2, 6]`
- band logits：`[B, T, 2, 3]`
- phase logits：`[B, T, 2, 4]`

## 标签生成规则摘要

- target_part: 根据 hand->part 距离 top1 最小值；若最小距离 >= `tau_near` 则为 `none`。
- hysteresis: 若上一帧目标距离不比当前最优差超过 `delta_target`，保持上一帧。
- band: `d < tau_contact` -> contact；`tau_contact <= d < tau_near` -> near；否则 far。
- phase: 基于距离变化与最近窗口的 near/contact 状态划分 idle/approach/hold/release。
- active: phase 为 approach/hold/release 时为 1。

默认超参：`tau_contact=0.10`，`tau_near=0.18`，`delta_target=0.02`，`epsilon_move=0.01`，`epsilon_hold=0.005`，`recent_window=3`。

## 最小可运行示例

```python
import torch
from model.contact.proposal_features import HandContactFeatureBuilder
from model.contact.proposal_labels import HandContactLabelBuilder
from model.contact.proposal_model import HandContactProposal
from model.contact.proposal_loss import HandContactProposalLoss

B, J, T = 2, 56, 60
actor_motion = torch.randn(B, J, 6, T)
coarse_motion = torch.randn(B, J, 6, T)
gt_motion = torch.randn(B, J, 6, T)
lengths = torch.full((B,), T, dtype=torch.long)

feat_builder = HandContactFeatureBuilder(device=actor_motion.device)
hand_feat, part_feat, rel_feat = feat_builder.build(actor_motion, coarse_motion, lengths=lengths)

model = HandContactProposal(hand_feat.shape[-1], part_feat.shape[-1])
logits = model(hand_feat, part_feat, rel_feat)

label_builder = HandContactLabelBuilder(device=actor_motion.device)
labels = label_builder.build(actor_motion, gt_motion, lengths=lengths)

criterion = HandContactProposalLoss()
loss, loss_dict = criterion(logits, labels, lengths=lengths)
```

## 事件解析与窗口构建示例

```python
import torch
from model.contact.proposal_events import parse_contact_events
from model.contact.proposal_windows import ContactWindowBuilder

active = torch.tensor([[[1,0],[1,0],[1,0],[0,0],[0,0],[0,0]]], dtype=torch.float)
target = torch.tensor([[[1,0],[1,0],[1,0],[0,0],[0,0],[0,0]]])
band = torch.tensor([[[2,0],[2,0],[1,0],[0,0],[0,0],[0,0]]])
phase = torch.tensor([[[2,0],[2,0],[1,0],[0,0],[0,0],[0,0]]])

events = parse_contact_events(active, target, band, phase)
window_builder = ContactWindowBuilder(window_size=8, pad=2)
windows = window_builder.build(events, lengths=[6])
mask = window_builder.to_mask(windows, lengths=[6])
```

## 设计假设

- 关节索引遵循当前项目使用的 SMPL-X joint 顺序。
- 手指 base/tip 索引用于手型摘要，如有不同关节定义需调整 `contact_defs.py`。

## 输出与集成建议

- Proposal 输出建议用于离线生成 hand-level 接触事件，再交给后续 `HandContactRefiner`。
- 若需要按事件窗口执行 refiner，可先 `parse_contact_events` -> `ContactWindowBuilder`。


## Refiner 简述

- `HandContactRefiner` 使用窗口内的局部手部序列、目标 patch 与关系特征进行精修。
- 训练时默认使用 teacher windows（由 GT 标签构造），支持 `window_source=predicted/mixed` 切换。
