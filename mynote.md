# ReGenNet Project Structure
```bash
ReGenNet
├── actor-x                  # ACTOR repo 动作识别模型
├── assets                   # 资源文件
├── body_models              # 人体模型参数，SMPL model and SMPL-X model
├── data_loaders             # 数据加载与数据管道代码
├── dataset                  # 数据集目录（原始/处理后）
├── diffusion                # 扩散模型相关实现
├── docker                   # Docker 构建与运行配置
├── eval                     # 评估脚本与指标
├── model                    # 核心模型定义
├── outputs                  # 运行输出与结果
├── preprocess               # 数据预处理脚本
├── recognition_training     # 训练好的动作识别模型，用于eval
├── render                   # 渲染与可视化渲染
├── sample                   # 示例脚本或样例输入
├── save                     # 模型权重与检查点
├── train                    # 训练脚本与配置
├── utils                    # 通用工具函数
├── visualize                # 可视化工具
├── cog.yaml                 # Cog 部署/推理配置
├── commands.txt             # keeewl常用命令或实验记录
├── environment5090.yml      # 特定硬件环境的 Conda 配置
├── environment.yml          # regennet conda env
├── LICENSE                  # 开源协议
├── mynote.md                # keeewl笔记
├── myReadMe.md              # keeewl记录
└── README.md                # regennet项目主说明
```


# Data preprocess
prepare_data.py → actor_reactor.py → split_2p.py。
