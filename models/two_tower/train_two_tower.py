import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from config.config_loader import get_config
from utils.logger import get_logger
from utils.seed import set_seed
from utils.device import get_device
from models.two_tower.model import TwoTowerModel
from models.two_tower.dataset import TwoTowerDataset, collate_fn
from evaluation.metrics import evaluate_vectorized


logger = get_logger('train_two_tower')


CF_BASELINE_HR10 = 0.0830  # UserUserCF K=50 from Section 3


def build_full_item_tensors(content_emb, genre_matrix, device):
    all_ids = torch.arange(content_emb.shape[0], dtype=torch.long, device=device)
    all_content = torch.tensor(content_emb, dtype=torch.float32, device=device)
    all_genres = torch.tensor(genre_matrix, dtype=torch.float32, device=device)
    return all_ids, all_content, all_genres


def _build_user_history_lookup(train_df, n_users, max_seq_len):
    """
    Precompute user_id -> padded history tensor + length array.
    Avoids per-user DataFrame filtering during eval (was 6B comparisons / eval).
    Returns:
        hist_arr: (n_users, max_seq_len) int64 left-padded with 0
        len_arr:  (n_users,) int64 — actual history lengths (>=1 for users with no train data)
    """
    hist_arr = np.zeros((n_users, max_seq_len), dtype=np.int64)
    len_arr = np.ones(n_users, dtype=np.int64)
    grouped = (
        train_df.sort_values('timestamp')
        .groupby('user_id', sort=False)['item_id']
        .apply(lambda s: s.tolist()[-max_seq_len:])
    )
    for uid, items in grouped.items():
        if not items:
            continue
        hist_arr[uid, max_seq_len - len(items):] = items
        len_arr[uid] = len(items)
    return hist_arr, len_arr


@torch.no_grad()
def get_score_matrix(model, all_item_ids, all_content, all_genres,
                      user_meta_df, train_df, device, batch_size=512, max_seq_len=200):
    """Compute (n_users, n_items) score matrix for vectorized eval."""
    model.eval()
    item_embs = model.encode_items(all_item_ids, all_content, all_genres)  # (n_items, 128)
    item_embs_cpu = item_embs.cpu()

    n_users = max(int(train_df['user_id'].max()), int(user_meta_df['user_id'].max())) + 1
    hist_arr, len_arr = _build_user_history_lookup(train_df, n_users, max_seq_len)

    demo = user_meta_df.set_index('user_id')[['gender_enc', 'age_enc', 'occupation']]
    demo_arr = demo.reindex(range(n_users)).fillna(0).values.astype(np.float32)

    user_embs = []
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        uids = torch.arange(start, end, dtype=torch.long, device=device)
        hist_t = torch.tensor(hist_arr[start:end], dtype=torch.long, device=device)
        lens_t = torch.tensor(len_arr[start:end], dtype=torch.long)
        demo_t = torch.tensor(demo_arr[start:end], dtype=torch.float32, device=device)
        u_emb = model.encode_users(uids, hist_t, lens_t, demo_t)
        user_embs.append(u_emb.cpu())

    user_embs = torch.cat(user_embs, dim=0)
    score_matrix = (user_embs @ item_embs_cpu.T).numpy().astype(np.float32)
    return score_matrix


def infonce_loss(logits):
    labels = torch.arange(logits.shape[0], device=logits.device)
    return nn.CrossEntropyLoss()(logits, labels)


def run():
    set_seed()
    cfg = get_config()
    device = get_device()
    processed = Path(cfg['data']['processed_dir'])
    tt_dir = Path('artifacts/two_tower')
    models_dir = Path('artifacts/models/two_tower')
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(processed / 'train.parquet')
    val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
    user_meta = pd.read_parquet(processed / 'user_metadata.parquet')
    stats = json.load(open(processed / 'dataset_stats.json'))
    n_users = stats['n_users']
    n_items = stats['n_items']

    content_emb = np.load(tt_dir / 'item_content_emb.npy')
    genre_matrix = np.load(tt_dir / 'item_genre_matrix.npy')

    tt_cfg = cfg['models']['two_tower']

    dataset = TwoTowerDataset(train_df, user_meta, max_seq_len=tt_cfg['max_seq_len'])
    loader = DataLoader(
        dataset, batch_size=tt_cfg['batch_size'],
        shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )
    logger.info(f'Dataset size: {len(dataset)}, batches: {len(loader)}')

    model = TwoTowerModel(n_users, n_items, temperature=tt_cfg['temperature']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tt_cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tt_cfg['epochs'])

    all_item_ids, all_content, all_genres = build_full_item_tensors(content_emb, genre_matrix, device)

    import wandb
    wandb.init(
        project=cfg['wandb']['project'],
        group='two_tower',
        name='two_tower_fixed',
        config={
            'model': 'TwoTower',
            'temperature': tt_cfg['temperature'],
            'batch_size': tt_cfg['batch_size'],
            'lr': tt_cfg['lr'],
            'epochs': tt_cfg['epochs'],
            'architecture': 'itemID+content+genre / userID+GRU+demographics',
        },
    )

    best_hr10 = 0.0
    patience_count = 0
    patience = tt_cfg['patience']

    for epoch in range(1, tt_cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            uids, iids, hists, lens, demos = [x.to(device) for x in batch]
            b_content = all_content[iids]
            b_genres = all_genres[iids]
            logits, _, _ = model(uids, hists, lens.cpu(), demos, iids, b_content, b_genres)
            loss = infonce_loss(logits)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)

        log_dict = {'epoch': epoch, 'train_loss': avg_loss, 'lr': scheduler.get_last_lr()[0]}

        if epoch % 5 == 0 or epoch == 1:
            score_matrix = get_score_matrix(
                model, all_item_ids, all_content, all_genres,
                user_meta, train_df, device, max_seq_len=tt_cfg['max_seq_len'],
            )
            metrics = evaluate_vectorized(score_matrix, val_df, train_df[['user_id', 'item_id']])
            log_dict.update(metrics)
            logger.info(f'Epoch {epoch}: loss={avg_loss:.4f} | HR@10={metrics["HR@10"]:.4f} NDCG@10={metrics["NDCG@10"]:.4f}')

            if metrics['HR@10'] > best_hr10:
                best_hr10 = metrics['HR@10']
                patience_count = 0
                torch.save(model.state_dict(), models_dir / 'two_tower_best.pt')
            else:
                patience_count += 1
                if patience_count >= patience:
                    logger.info(f'Early stopping at epoch {epoch}')
                    wandb.log(log_dict)
                    break

        wandb.log(log_dict)

    wandb.log({'best_HR@10': best_hr10, 'cf_baseline_HR@10': CF_BASELINE_HR10})
    wandb.finish()
    logger.info(f'Section 5 complete. Best HR@10={best_hr10:.4f}')


if __name__ == '__main__':
    from models.two_tower.feature_prep import build_item_feature_matrices
    build_item_feature_matrices(get_config())
    run()
