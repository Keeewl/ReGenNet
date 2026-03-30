## ReGenNet 仓库整理记录 (Chi3D + InterX)

目标
- 保留 chi3d + interx / 双人 / online / offline / text 条件 / unconstrained
- 保留 actor-x 自训流程与 cnet_v*
- 保证 commands.txt 里的命令可用
- 清理 NTU / HumanML / HumanAct12 / UESTC / rot_vel 分支等无关逻辑


## 已完成的整理内容

### data_loaders/
- 仅保留 chi3d/interx 数据集入口与读写逻辑
- 删除 NTU/UESTC/HumanAct12 相关 dataset 文件
- 删除 HumanML 数据/评估工具模块 (data_loaders/humanml 全部移除)

### diffusion/
- gaussian_diffusion.py:
  - 仅保留 chi3d dataset 判断
  - 删除 foot_contact_loss_humanml3d 与 velocity_consistency_loss_humanml3d
  - 移除对应引用
  - 删除 rot_vel 相关 loss 参数入口
- 其他 diffusion 文件保留不变

### eval/
- 保留 chi3d/interx 评估链路:
  - eval/eval_cmdm.py
  - eval/easy_table.py
  - eval/a2m/stgcn_eval.py
  - eval/a2m/stgcn/*
  - eval/a2m/recognition/models/*
  - eval/a2m/tools.py
  - eval/a2m/__init__.py
- 删除 humanml/humanact12/uestc/GRU/A2M 相关评估分支
- train/training_loop.py 中 HumanML eval 分支已清理
- actor-x 评估仅保留 stgcn (chi3d/interx)

### sample/
- 删除:
  - sample/edit.py
  - sample/predict.py
- 保留:
  - sample/cgenerate.py (commands.txt 依赖)
  - 删除 humanml/hml_vec 采样分支，text-only 保留

### preprocess/
- 删除 NTU 预处理脚本
- 新建 preprocess/chi3d/README.md 占位
- 新增 preprocess/interx/ 预处理脚本 (手动添加)

### utils/
- parser_util.py:
  - 删除 edit 相关参数解析
  - 数据集选项保留 chi3d/interx
  - 保留 unconstrained / text prompt / t2m 相关选项
- model_util.py:
  - 删除 humanml/kit/ntu 分支
  - 固定 chi3d 参数 (num_frames=150)

### train/ model/ render/
- cmdm + cnet_v* 去掉 rot_vel 分支
- 其他保持不动 (按需求保留)


## 当前目录变更速览
- 删除: NTU / HumanML / HumanAct12 / UESTC / GRU / A2M 相关脚本与评估链路
- 删除: rot_vel 相关模型与 loss 分支
- 新增: preprocess/chi3d/README.md 占位
- 新增: preprocess/interx/ 预处理脚本


## 仍保留的核心运行路径
- 训练: train/train_mdm.py
- 采样: sample/cgenerate.py
- 渲染: render/crendermotion.py
- 评估: eval/eval_cmdm.py + eval/a2m/stgcn_eval.py
- 数据: data_loaders/a2m/feeder.py + data_loaders/get_data.py


## Chi3D 预处理说明
- 仓库内无 chi3d 预处理脚本
- 当前使用作者提供的 chi3d .h5
- preprocess/chi3d/README.md 仅为占位

## InterX 预处理说明
- preprocess/interx/ 为手动添加的预处理脚本
