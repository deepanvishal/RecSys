import numpy as np
import torch
from torch.utils.data import Dataset


class BERT4RecDataset(Dataset):
    """
    One sequence per user. Each __getitem__ resamples a fresh random Cloze mask:
    ~mask_prob fraction of items become MASK, label = original item id (0-indexed).
    Non-masked positions get label=-100 to be ignored by CE loss.
    """
    def __init__(self, train_df, n_items, max_len=200, mask_prob=0.2):
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = n_items + 1
        self.n_items = n_items
        self.sequences = []
        df = train_df.sort_values(['user_id', 'timestamp'])
        for _, group in df.groupby('user_id', sort=False):
            items = group['item_id'].tolist()
            shifted = [i + 1 for i in items]
            seq = shifted[-max_len:]
            self.sequences.append(seq)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        masked_seq, labels = self._mask(seq)
        pad_len = self.max_len - len(masked_seq)
        masked_padded = [0] * pad_len + masked_seq
        labels_padded = [-100] * pad_len + labels
        return (
            torch.tensor(masked_padded, dtype=torch.long),
            torch.tensor(labels_padded, dtype=torch.long),
        )

    def _mask(self, seq):
        masked, labels = [], []
        for item in seq:
            if np.random.random() < self.mask_prob:
                masked.append(self.mask_token)
                labels.append(item - 1)  # 0-indexed for CE loss
            else:
                masked.append(item)
                labels.append(-100)
        return masked, labels


def build_user_sequences_bert(train_df, n_items, max_len=200):
    """
    Inference sequences: keep last (max_len-1) train items, append MASK at the end.
    Returns dict: user_id -> padded sequence (length max_len, MASK at position -1).
    """
    mask_token = n_items + 1
    seqs = {}
    df = train_df.sort_values(['user_id', 'timestamp'])
    for uid, group in df.groupby('user_id', sort=False):
        items = group['item_id'].tolist()
        shifted = [i + 1 for i in items]
        seq = shifted[-(max_len - 1):]
        seq = seq + [mask_token]
        padded = [0] * (max_len - len(seq)) + seq
        seqs[int(uid)] = padded
    return seqs
