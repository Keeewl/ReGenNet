"""Data loading helpers for refine_v2 CLIs."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from refine.data import ReactionDataDataset, reaction_data_collate


class IndexedReactionDataDataset(Dataset):
    """Thin wrapper adding dataset_row_index to existing reaction_data samples."""

    def __init__(self, reaction_data_path: str):
        self.base = ReactionDataDataset(reaction_data_path)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        item["dataset_row_index"] = int(index)
        return item

    def close(self):
        self.base.close()


def make_reaction_data_loader(
    reaction_data_path: str,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    dataset = IndexedReactionDataDataset(reaction_data_path)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=reaction_data_collate,
    )

