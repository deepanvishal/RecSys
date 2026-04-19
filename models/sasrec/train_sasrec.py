import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from config.config_loader import get_config
from utils.logger import get_logger
from utils.seed import set_seed
from utils.device import get_device
from models.sasrec.model import SASRec
from models.sasrec.dataset import SASRecDataset, build_user_sequences
from evaluation.metrics import evaluate_vectorized


logger = get_logger('train_sasrec')


SVD_BASELINE_HR10 = 0.1020
CF_BASELINE_HR10 = 0.0830


@torch.no_grad()
def get_score_matrix(model, user_seqs, n_users, n_items, device, batch_size=512):
    """Encode each user once -> (n_users, n_items) score matrix."""
    model.eval()
    user_ids = sorted(user_seqs.keys())
    score_matrix = np.zeros((n_users, n_items), dtype=np.float32)
    for start in range(0, len(user_ids), batch_size):
        batch_ids = user_ids[start:start + batch_size]
        seqs = torch.tensor(
            [user_seqs[uid] for uid in batch_ids],
            dtype=torch.long, device=device,
        )
        scores = model.score_all_items(seqs).cpu().numpy()
        for i, uid in enumerate(batch_ids):
            score_matrix[uid] = scores[i]
    return score_matrix


def run():
    set_seed()
    cfg = get_config()
    device = get_device()
    processed = Path(cfg['data']['processed_dir'])
    models_dir = Path('artifacts/models/sasrec')
    models_dir.mkdir(parents=True, exist_ok=True)

    stats = json.load(open(processed / 'dataset_stats.json'))
    n_users = stats['n_users']
    n_items = stats['n_items']

    train_df = pd.read_parquet(processed / 'train.parquet')
    val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
    train_pairs = train_df[['user_id', 'item_id']]

    sr_cfg = cfg['models']['sasrec']

    dataset = SASRecDataset(train_df, n_items, max_len=sr_cfg['max_len'])
    loader = DataLoader(
        dataset, batch_size=sr_cfg['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True,
    )
    logger.info(f'SASRec dataset: {len(dataset)} samples, {len(loader)} batches')

    model = SASRec(
        n_items=n_items,
        hidden=sr_cfg['hidden'],
        max_len=sr_cfg['max_len'],
        num_blocks=sr_cfg['num_blocks'],
        num_heads=sr_cfg['num_heads'],
        dropout=sr_cfg['dropout'],
    ).to(device)
    logger.info(f'SASRec params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=sr_cfg['lr'], weight_decay=sr_cfg['weight_decay'],
    )
    criterion = nn.CrossEntropyLoss()

    user_seqs = build_user_sequences(train_df, max_len=sr_cfg['max_len'])

    wandb.init(
        project=cfg['wandb']['project'],
        group='sasrec',
        name='sasrec_plus',
        config={**sr_cfg, 'loss': 'CE', 'n_items': n_items, 'n_users': n_users},
    )

    best_hr10 = 0.0
    patience_count = 0

    for epoch in range(1, sr_cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for seqs, targets in loader:
            seqs, targets = seqs.to(device), targets.to(device)
            logits = model.score_all_items(seqs)   # (B, n_items)
            loss = criterion(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        log_dict = {'epoch': epoch, 'train_loss': avg_loss}

        score_matrix = get_score_matrix(model, user_seqs, n_users, n_items, device)
        metrics = evaluate_vectorized(score_matrix, val_df, train_pairs)
        log_dict.update(metrics)
        logger.info(f'Epoch {epoch:3d}: loss={avg_loss:.4f} | HR@10={metrics["HR@10"]:.4f} NDCG@10={metrics["NDCG@10"]:.4f}')

        if metrics['HR@10'] > best_hr10:
            best_hr10 = metrics['HR@10']
            patience_count = 0
            torch.save(model.state_dict(), models_dir / 'sasrec_best.pt')
        else:
            patience_count += 1
            if patience_count >= sr_cfg['patience']:
                logger.info(f'Early stopping at epoch {epoch}')
                wandb.log(log_dict)
                break

        wandb.log(log_dict)

    wandb.log({
        'best_HR@10': best_hr10,
        'svd_baseline_HR@10': SVD_BASELINE_HR10,
        'cf_baseline_HR@10': CF_BASELINE_HR10,
    })
    wandb.finish()
    logger.info(f'Section 6 SASRec complete. Best HR@10={best_hr10:.4f}')


if __name__ == '__main__':
    run()
