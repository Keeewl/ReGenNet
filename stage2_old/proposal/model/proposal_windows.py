import torch

from stage2_old.common.geometry.contact_defs import HAND_SIDES


class ContactWindowBuilder:
    def __init__(self, window_size=None, pad=0):
        self.window_size = window_size
        self.pad = int(pad)

    def build(self, events, lengths=None):
        """
        events: list(list(dict)) per batch
        returns list(list(dict)) with windowed segments
        """
        results = []
        if torch.is_tensor(lengths):
            lengths = lengths.detach().cpu().tolist()
        for b, items in enumerate(events):
            length = None if lengths is None else int(lengths[b])
            windows = []
            for event in items:
                start = int(event["start_frame"])
                end = int(event["end_frame"])
                if self.pad > 0:
                    start = max(0, start - self.pad)
                    end = end + self.pad if length is None else min(length - 1, end + self.pad)
                if self.window_size is not None:
                    win = int(self.window_size)
                    center = (start + end) // 2
                    start = max(0, center - win // 2)
                    end = start + win - 1
                    if length is not None:
                        if end >= length:
                            end = length - 1
                            start = max(0, end - win + 1)
                window = dict(event)
                window["start_frame"] = int(start)
                window["end_frame"] = int(end)
                windows.append(window)
            results.append(windows)
        return results

    def to_mask(self, windows, lengths, num_hands=2):
        """
        windows: list(list(dict)) per batch
        returns mask: [B, T, 2] bool
        """
        if torch.is_tensor(lengths):
            lengths = lengths.detach().cpu().tolist()
        batch_size = len(windows)
        max_len = int(max(lengths))
        mask = torch.zeros(batch_size, max_len, num_hands, dtype=torch.bool)
        side_to_idx = {name: idx for idx, name in enumerate(HAND_SIDES)}
        for b, items in enumerate(windows):
            length = int(lengths[b])
            for window in items:
                side = window.get("hand_side", "left")
                h = side_to_idx.get(side, 0)
                start = max(0, int(window["start_frame"]))
                end = min(length - 1, int(window["end_frame"]))
                if start <= end:
                    mask[b, start : end + 1, h] = True
        return mask
