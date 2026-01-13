## Installation
Done:
1. clone repo ✅

2. setup the environment (maybe) ✅ 以后都用regennet5090这个conda环境 ✅

3. Download other required files
- 把 pretrained models 放到 save/ 目录下 ✅
- 下载 action recognition models 放到 recognition_training/ 目录下 ✅ 用来 evaluation ❌（acc0.15）
- 下载 SMPL + SMPL-X models 放到 body_models/ 目录下 ✅

Todo:
- [x] 用 action recognition models 来复现 evaluation：第一轮acc0.2-0.3完全不对，debug ⌛️✅
- [x] debug eval 用自己训练的动作识别模型，基本能复现eval指标


## Data Preparation
Done:
1. Chi3D 
- 已下载 chi3d 的原始数据集，存放在 my-HHI/ ✅ 用来跑实验 ✅
- 已下载 actor-reactor order annotations 放到 dataset/chi3d/annotations_chi3d ✅ 
- 已下载 processed dataset (.h5 files) 放到 dataset/chi3d ✅

Todo:
- [] 用 chi3d 原始数据集跑实验


## Train
完全没跑过 ⌛️


## Evaluation
用自己训练的动作识别模型，基本复现eval指标：
-   **fid_gen_train**：0.21 ± 0.0092
-   **accuracy_gen_train**：0.996 ± 0.00
-   **multimodality_gen_train**：6.226 ± 0.1547
-   **diversity_gen_train**：15.868 ± 0.2431
-   **fid_gen_test**：21.731 ± 13.4312
-   **accuracy_gen_test**：0.365 ± 0.0031
-   **multimodality_gen_test**：6.257 ± 1.019
-   **diversity_gen_test**：9.493 ± 1.6995


## Motion Synthesis and Visualize
Done:
1. 现在用regennet5090这个环境 ✅
2. 在 chi3d_smplx_test.h5 跑 chi3d 的 generate 和 render，效果不好 ⌛️
3. 在 chi3d_smplx_train.h5 跑 chi3d 的 generate 和 render，效果不好 ⌛️ (2,3 commands in commands.txt)
