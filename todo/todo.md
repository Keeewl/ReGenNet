## 我想要实现的主框架/功能

-[] 预期最后的论文/报告是有两个大表：一个是stage1的STGCN指标，一个是stage2的contact相关指标（stage2主要的优化目标），可选stage2同意输出STGCN的指标，来对比GT/coarse/refined的STGCN指标（期望是在主要优化contact的同时，STGCN指标不明显降低）

-[] 后期做一个把stage1和stage2整合起来的采样脚本（类似stage1的），能够输出GT/coarse/refined的动作序列outputs，可以让我可视化验证，同时计算infer的时间消耗（期望refiner是轻量高效的）

-[] 接入baseline与stage2的接口，这个stage2是一个“高效可插拔”模块/阶段，也能够显著改善baseline的指标和效果。

-[] 