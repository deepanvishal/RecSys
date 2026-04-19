import numpy as np
import torch
from torch.utils.data import Dataset


class SASRecDataset(Dataset):
    """
    Each sample = (left-padded prefix sequence, target item).
    Item IDs shifted by +1 in the sequence so that 0 is reserved for padding.
    Target is 0-indexed for CE loss against (n_items)-class logits.
    Sequences and targets stored as numpy arrays — DataLoader workers share via fork COW.
    """
    def __init__(self, train_df, n_items, max_len=200):
        self.max_len = max_len
        self.n_items = n_items
        df = train_df.sort_values(['user_id', 'timestamp'])
        seq_buf = []
        tgt_buf = []
        for _, group in df.groupby('user_id', sort=False):
            items = group['item_id'].tolist()
            shifted = [i + 1 for i in items]
            for t in range(1, len(shifted)):
                prefix = shifted[:t][-max_len:]
                padded = [0] * (max_len - len(prefix)) + prefix
                seq_buf.append(padded)
                tgt_buf.append(items[t])  # 0-indexed item id
        self.seqs = np.asarray(seq_buf, dtype=np.int64)        # (N, max_len)
        self.targets = np.asarray(tgt_buf, dtype=np.int64)     # (N,)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.seqs[idx]),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )


def build_user_sequences(train_df, max_len=200):
    """
    Build one padded sequence per user (for evaluation).
    Returns dict: user_id -> list[int] (length max_len, left-padded with 0,
    item ids shifted +1).
    """
    seqs = {}
    df = train_df.sort_values(['user_id', 'timestamp'])
    for uid, group in df.groupby('user_id', sort=False):
        items = group['item_id'].tolist()
        shifted = [i + 1 for i in items]
        seq = shifted[-max_len:]
        padded = [0] * (max_len - len(seq)) + seq
        seqs[int(uid)] = padded
    return seqs
