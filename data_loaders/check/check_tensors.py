"""
tensors.py 实现的功能总结（在训练数据流中的位置）

data_loaders/tensors.py 的定位：DataLoader 的 collate_fn 实现。
PyTorch 的 Dataset.__getitem__() 通常返回“单样本”（可能长度不一、字段不一），
而 DataLoader 需要把一批样本拼成一个 batch。这个“拼 batch”的逻辑就放在 collate_fn 里。

tensors.py 提供了三套 collate：
collate(batch)：通用 MDM/非双人条件
输入：batch = [sample_dict1, sample_dict2, ...]
输出：motion, cond
做的事：padding、生成 mask、收集 text/tokens/action 等条件

ccollate(batch)：CMDM 专用（双人条件生成）
假设单样本的 inp 是 [J, Dp, T]，其中 Dp=2*D（两个人拼在 feature 维）
它把 feature 维一刀切成两半：
cmotion（条件人）: [J, D, T]
motion（目标人）: [J, D, T]
然后对两者分别 padding 成 batch，并把 cmotion 放进 cond['y']

t2m_collate(batch)：HumanML/Kit 文本-动作数据的适配器
把 humanml 的 tuple 样本格式转成 dict，然后复用 collate()

此外，还有两个底层工具函数：
lengths_to_mask(lengths, max_len)：把长度向量变成 padding mask
collate_tensors(batch)：把一组不同 shape 的 tensor padding 成一个 batch tensor（按每维最大值补零）

一句话总结：
tensors.py 负责把“长度不一、字段可能不同的单样本”变成“统一张量 + mask + 条件字典”，让后面的 model/diffusion/loss 可以无脑吃 [B, ...] 的数据。
"""
