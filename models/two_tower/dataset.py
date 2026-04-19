import torch
from torch.utils.data import Dataset


class TwoTowerDataset(Dataset):
    """
    Each sample = (user_id, target_item_id, history_sequence, hist_len, demographics).
    History = all items the user interacted with strictly before the target,
    sorted by timestamp, capped at max_seq_len (kept right-most = most recent).
    First interaction per user is dropped (no history to learn from).
    """
    def __init__(self, interactions_df, user_meta_df, max_seq_len=200):
        self.max_seq_len = max_seq_len
        df = interactions_df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        # Build (history, target) samples — explicit loop is faster + safer than groupby.transform
        user_ids, item_ids, histories = [], [], []
        for uid, group in df.groupby('user_id', sort=False):
            items = group['item_id'].tolist()
            for i in range(1, len(items)):
                user_ids.append(uid)
                item_ids.append(items[i])
                histories.append(items[:i])
        self.user_ids = torch.tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.tensor(item_ids, dtype=torch.long)
        self.histories = histories

        # Demographics lookup aligned to user_id range [0, max_user_id]
        max_uid = max(int(df['user_id'].max()), int(user_meta_df['user_id'].max()))
        demo = user_meta_df.set_index('user_id')[['gender_enc', 'age_enc', 'occupation']]
        demo_arr = demo.reindex(range(max_uid + 1)).fillna(0).values
        self.demo = torch.tensor(demo_arr, dtype=torch.float32)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        uid = self.user_ids[idx]
        iid = self.item_ids[idx]
        hist = self.histories[idx][-self.max_seq_len:]
        hist_len = len(hist)
        padded = [0] * (self.max_seq_len - hist_len) + hist
        hist_tensor = torch.tensor(padded, dtype=torch.long)
        demo = self.demo[uid]
        return uid, iid, hist_tensor, torch.tensor(hist_len, dtype=torch.long), demo


def collate_fn(batch):
    uids, iids, hists, lens, demos = zip(*batch)
    return (
        torch.stack(uids),
        torch.stack(iids),
        torch.stack(hists),
        torch.stack(lens),
        torch.stack(demos),
    )
