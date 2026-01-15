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


import torch

from data_loaders.tensors import (
    lengths_to_mask,
    collate_tensors,
    collate,
    ccollate,
    t2m_collate,
)


def _print_tensor(label, tensor):
    print(f"{label}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    print(tensor)


def test_lengths_to_mask():
    print("== lengths_to_mask ==")
    lengths = torch.tensor([2, 4, 1])
    mask = lengths_to_mask(lengths, max_len=5)
    _print_tensor("lengths", lengths)
    _print_tensor("mask", mask)
    assert mask.shape == (3, 5)
    assert mask[0].tolist() == [True, True, False, False, False]
    assert mask[1].tolist() == [True, True, True, True, False]
    assert mask[2].tolist() == [True, False, False, False, False]


def test_collate_tensors():
    print("== collate_tensors ==")
    b1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
    b3 = torch.tensor([[13, 14, 15, 16]])
    out = collate_tensors([b1, b2, b3])
    _print_tensor("b1", b1)
    _print_tensor("b2", b2)
    _print_tensor("b3", b3)
    _print_tensor("out", out)
    assert out.shape == (3, 3, 4)
    assert torch.all(out[0, :2, :3] == b1)
    assert torch.all(out[1, :3, :2] == b2)
    assert torch.all(out[2, :1, :4] == b3)
    assert torch.all(out[0, 2:, :] == 0)


def test_collate_with_conditions():
    print("== collate (with conditions) ==")
    sample1 = {
        "inp": torch.arange(2 * 3 * 4).reshape(2, 3, 4),
        "lengths": 4,
        "text": "walk forward",
        "tokens": ["walk", "forward"],
        "action": 1,
        "action_text": "walk",
    }
    sample2 = {
        "inp": torch.arange(2 * 3 * 2).reshape(2, 3, 2),
        "lengths": 2,
        "text": "jump",
        "tokens": ["jump"],
        "action": 2,
        "action_text": "jump",
    }
    motion, cond = collate([sample1, sample2])
    _print_tensor("sample1 inp", sample1["inp"])
    _print_tensor("sample2 inp", sample2["inp"])
    _print_tensor("motion", motion)
    _print_tensor("mask", cond["y"]["mask"])
    _print_tensor("lengths", cond["y"]["lengths"])
    print("text:", cond["y"]["text"])
    print("tokens:", cond["y"]["tokens"])
    print("action:", cond["y"]["action"].squeeze(1).tolist())
    print("action_text:", cond["y"]["action_text"])
    assert motion.shape == (2, 2, 3, 4)
    assert cond["y"]["mask"].shape == (2, 1, 1, 4)
    assert cond["y"]["lengths"].tolist() == [4, 2]


def test_collate_without_lengths():
    print("== collate (no lengths) ==")
    sample1 = {"inp": torch.ones(1, 2, 3)}
    sample2 = {"inp": torch.ones(1, 2, 2)}
    motion, cond = collate([sample1, sample2])
    _print_tensor("motion", motion)
    _print_tensor("mask", cond["y"]["mask"])
    assert cond["y"]["lengths"].tolist() == [3, 2]
    assert motion.shape == (2, 1, 2, 3)


def test_ccollate():
    print("== ccollate (two-person split) ==")
    # [J, Dp, T] where Dp = 2 * D
    sample1 = {
        "inp": torch.arange(2 * 4 * 3).reshape(2, 4, 3),
        "lengths": 3,
        "text": "pair dance",
    }
    sample2 = {
        "inp": torch.arange(2 * 4 * 2).reshape(2, 4, 2),
        "lengths": 2,
        "text": "pair jump",
    }
    motion, cond = ccollate([sample1, sample2])
    _print_tensor("sample1 inp", sample1["inp"])
    _print_tensor("sample2 inp", sample2["inp"])
    _print_tensor("motion", motion)
    _print_tensor("cmotion", cond["y"]["cmotion"])
    _print_tensor("mask", cond["y"]["mask"])
    print("text:", cond["y"]["text"])
    assert motion.shape == (2, 2, 2, 3)
    assert cond["y"]["cmotion"].shape == (2, 2, 2, 3)
    assert cond["y"]["mask"].shape == (2, 1, 1, 3)


def test_t2m_collate():
    print("== t2m_collate (humanml adapter) ==")
    # tuple format: (..., text at [2], motion at [4], length at [5], tokens at [6])
    sample1 = (None, None, "turn left", None, [[1, 2], [3, 4], [5, 6]], 3, ["turn", "left"])
    sample2 = (None, None, "sit down", None, [[7, 8], [9, 10]], 2, ["sit", "down"])
    motion, cond = t2m_collate([sample1, sample2])
    _print_tensor("motion", motion)
    _print_tensor("mask", cond["y"]["mask"])
    print("text:", cond["y"]["text"])
    print("tokens:", cond["y"]["tokens"])
    assert motion.shape == (2, 2, 1, 3)
    assert cond["y"]["lengths"].tolist() == [3, 2]
    assert cond["y"]["mask"].shape == (2, 1, 1, 3)


def run_all_tests():
    test_lengths_to_mask()
    test_collate_tensors()
    test_collate_with_conditions()
    test_collate_without_lengths()
    test_ccollate()
    test_t2m_collate()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
