"""Finetuning dataset: reads prepped filelists (clip_id|lang_id|phonemes), loads cached mels,
normalizes with the BASE model's mel stats (keeps the model + vocoder in one mel space),
prepends the per-language token, and batches in the format MatchaTTS.forward expects.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from matcha.text import text_to_sequence
from matcha.text.symbols import lang_token_id
from matcha.utils.model import fix_len_compatibility
from matcha.utils.utils import intersperse


def parse_filelist(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) >= 3 and parts[0]:
                rows.append((parts[0], int(parts[1]), parts[2]))
    return rows


class FinetuneDataset(Dataset):
    def __init__(self, filelist, mels_dir, mel_mean, mel_std, add_blank=True):
        self.rows = parse_filelist(filelist)
        self.mels_dir = Path(mels_dir)
        self.mel_mean, self.mel_std = float(mel_mean), float(mel_std)
        self.add_blank = add_blank

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        clip_id, lid, phon = self.rows[i]
        seq, _ = text_to_sequence(phon, ["twi_phonemes"])  # passthrough (already phonemized)
        if self.add_blank:
            seq = intersperse(seq, 0)
        seq = [lang_token_id(lid)] + seq  # language token at the front
        x = torch.tensor(seq, dtype=torch.long)

        mel = torch.from_numpy(np.load(self.mels_dir / f"{clip_id}.npy").astype("float32"))
        mel = (mel - self.mel_mean) / self.mel_std  # normalize with BASE stats
        return {"x": x, "y": mel, "spk": lid}


class Collate:
    def __call__(self, batch):
        B = len(batch)
        x_max = max(b["x"].shape[-1] for b in batch)
        y_max = fix_len_compatibility(max(b["y"].shape[-1] for b in batch))
        n_feats = batch[0]["y"].shape[0]
        x = torch.zeros(B, x_max, dtype=torch.long)
        y = torch.zeros(B, n_feats, y_max, dtype=torch.float32)
        x_len, y_len, spks = [], [], []
        for i, b in enumerate(batch):
            x[i, : b["x"].shape[-1]] = b["x"]
            y[i, :, : b["y"].shape[-1]] = b["y"]
            x_len.append(b["x"].shape[-1]); y_len.append(b["y"].shape[-1]); spks.append(b["spk"])
        return {
            "x": x, "x_lengths": torch.tensor(x_len, dtype=torch.long),
            "y": y, "y_lengths": torch.tensor(y_len, dtype=torch.long),
            "spks": torch.tensor(spks, dtype=torch.long),
        }
