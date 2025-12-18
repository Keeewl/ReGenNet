## Installation
Done:
1. clone repo ✅

2. setup the environment (maybe) ✅

3. Download other required files

- 把 pretrained models 放到 save/ 目录下 ✅
- 下载 action recognition models 放到 recognition_training/ 目录下 ✅ 用来 evaluation ⌛️
- 下载 SMPL + SMPL-X models 放到 body_models/ 目录下 ✅

Todo:

- [] 用 action recognition models 来复现 evaluation？


## Data Preparation
Done:
1. Chi3D 

- 已下载 chi3d 的原始数据集，存放在 my-HHI/ ✅ 用来跑实验 ⌛️
- 已下载 actor-reactor order annotations 放到 dataset/chi3d/annotations_chi3d ✅ 
- 已下载 processed dataset (.h5 files) 放到 dataset/chi3d ✅

Todo:
- [] 用 chi3d 原始数据集跑实验？
- [] NTU RGB+D 120
- [] InterHuman


## Train
完全没跑过


## Evaluation
尝试跑指标


## Motion Synthesis and Visualize
Done:
1. 把 render 的环境搭好（未导出）（目前是regennet-render这个环境都能跑）⌛️
2. 在 chi3d_smplx_test.h5 跑 chi3d 的 generate 和 render，效果不好 ⌛️
3. 在 chi3d_smplx_train.h5 跑 chi3d 的 generate 和 render，效果不好 ⌛️ (2,3 commands in log.txt)

Todo:
- [] 跑指标 evaluation VS Debug