import torch

from stage2_old.common.geometry.contact_defs import BAND_IDS, HAND_SIDES, PHASE_IDS, TARGET_PARTS
from stage2_old.common.geometry.mesh_regions import WINDOW_STATE_IDS, WINDOW_STATE_NAMES


def logits_to_frame_labels(logits, active_threshold=0.5):
    active = (torch.sigmoid(logits["active"]).squeeze(-1) > float(active_threshold)).long()
    target = torch.argmax(logits["target"], dim=-1)
    band = torch.argmax(logits["band"], dim=-1)
    phase = torch.argmax(logits["phase"], dim=-1)
    return {
        "active": active,
        "target_part": target,
        "band": band,
        "phase": phase,
    }


class DiffusionWindowBuilder:
    """
    Build strict/near windows from blueprint frame labels.
    """

    def __init__(self, window_size=None, pad=0, max_window_size=None, min_window_size=1, dedup=True):
        self.window_size = window_size
        self.pad = int(pad)
        self.max_window_size = max_window_size
        self.min_window_size = int(min_window_size)
        self.dedup = bool(dedup)

    def _apply_window_bounds(self, start, end, length=None):
        if self.pad > 0:
            start = max(0, start - self.pad)
            if length is None:
                end = end + self.pad
            else:
                end = min(length - 1, end + self.pad)

        win_len = end - start + 1
        target_len = None
        if self.window_size is not None:
            target_len = int(self.window_size)
        elif self.max_window_size is not None and win_len > int(self.max_window_size):
            target_len = int(self.max_window_size)

        if target_len is not None:
            center = (start + end) // 2
            start = max(0, center - target_len // 2)
            end = start + target_len - 1
            if length is not None and end >= length:
                end = length - 1
                start = max(0, end - target_len + 1)
        return start, end

    def _scan_segments(self, mask, target_ids, length):
        segments = []
        start = None
        current = None
        for t in range(length):
            active = bool(mask[t]) and int(target_ids[t]) > 0
            if active:
                if start is None:
                    start = t
                    current = int(target_ids[t])
                elif int(target_ids[t]) != current:
                    segments.append((start, t - 1, current))
                    start = t
                    current = int(target_ids[t])
            else:
                if start is not None:
                    segments.append((start, t - 1, current))
                    start = None
                    current = None
        if start is not None:
            segments.append((start, length - 1, current))
        return segments

    def _build_windows(self, mask, target_ids, length, hand_side, state):
        windows = []
        segments = self._scan_segments(mask, target_ids, length)
        for start, end, target_id in segments:
            if end - start + 1 < self.min_window_size:
                continue
            start, end = self._apply_window_bounds(start, end, length=length)
            if end < start:
                continue
            target_name = TARGET_PARTS[target_id]
            windows.append(
                {
                    "hand_side": hand_side,
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "target_part_id": int(target_id),
                    "target_part": target_name,
                    "window_state": state,
                    "window_state_id": int(WINDOW_STATE_IDS[state]),
                }
            )
        return windows

    def expand_windows(self, windows, length, buffer_frames=0, dedup=None):
        buffer_frames = int(buffer_frames)
        if buffer_frames <= 0:
            return list(windows)
        dedup = self.dedup if dedup is None else bool(dedup)
        expanded = []
        seen = set()
        for win in windows:
            item = dict(win)
            item["start_frame"] = max(0, int(item["start_frame"]) - buffer_frames)
            item["end_frame"] = min(int(length) - 1, int(item["end_frame"]) + buffer_frames)
            if dedup:
                key = (
                    item["hand_side"],
                    item["start_frame"],
                    item["end_frame"],
                    item["target_part_id"],
                    item["window_state_id"],
                )
                if key in seen:
                    continue
                seen.add(key)
            expanded.append(item)
        return expanded

    def expand_windows_batch(self, windows_per_batch, lengths, buffer_frames=0, dedup=None):
        if buffer_frames <= 0:
            return windows_per_batch
        if torch.is_tensor(lengths):
            lengths = lengths.detach().cpu().tolist()
        out = []
        for items, length in zip(windows_per_batch, lengths):
            out.append(self.expand_windows(items, int(length), buffer_frames=buffer_frames, dedup=dedup))
        return out

    def build_from_labels(self, labels, lengths=None):
        """
        labels: dict with active/target_part/band/phase [B, T, 2]
        returns strict_windows, near_windows (list per batch)
        """
        target = labels["target_part"]
        band = labels["band"]
        phase = labels["phase"]

        if torch.is_tensor(lengths):
            lengths = lengths.detach().cpu().tolist()
        batch_size, num_frames, num_hands = target.shape

        strict_windows = []
        near_windows = []

        for b in range(batch_size):
            length = num_frames if lengths is None else int(lengths[b])
            strict_batch = []
            near_batch = []
            for h in range(num_hands):
                band_seq = band[b, :length, h]
                phase_seq = phase[b, :length, h]
                target_seq = target[b, :length, h]

                strict_mask = (band_seq == BAND_IDS["contact"]) | (phase_seq == PHASE_IDS["hold"])
                near_mask = (
                    (band_seq == BAND_IDS["near"])
                    | (phase_seq == PHASE_IDS["approach"])
                    | (phase_seq == PHASE_IDS["release"])
                )
                near_mask = near_mask & (~strict_mask)

                hand_side = HAND_SIDES[h]
                strict_batch.extend(self._build_windows(strict_mask, target_seq, length, hand_side, "strict"))
                near_batch.extend(self._build_windows(near_mask, target_seq, length, hand_side, "near"))

            if self.dedup:
                seen = set()
                deduped = []
                for win in strict_batch:
                    key = (win["hand_side"], win["start_frame"], win["end_frame"], win["target_part_id"], win["window_state_id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(win)
                strict_batch = deduped
                seen = set()
                deduped = []
                for win in near_batch:
                    key = (win["hand_side"], win["start_frame"], win["end_frame"], win["target_part_id"], win["window_state_id"])
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(win)
                near_batch = deduped

            strict_windows.append(strict_batch)
            near_windows.append(near_batch)

        return strict_windows, near_windows
