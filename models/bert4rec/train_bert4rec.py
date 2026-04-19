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
from models.bert4rec.model import BERT4Rec
from models.bert4rec.dataset import BERT4RecDataset, build_user_sequences_bert
from evaluation.metrics import evaluate_vectorized


logger = get_logger('train_bert4rec')


SASREC_HR10 = 0.3328
SVD_BASELINE_HR10 = 0.1020


@torch.no_grad()
def get_score_matrix(model, user_seqs, n_users, n_items, device, batch_size=512):
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
    models_dir = Path('artifacts/models/bert4rec')
    models_dir.mkdir(parents=True, exist_ok=True)

    stats = json.load(open(processed / 'dataset_stats.json'))
    n_users = stats['n_users']
    n_items = stats['n_items']

    train_df = pd.read_parquet(processed / 'train.parquet')
    val_df = pd.read_parquet(processed / 'val.parquet')[['user_id', 'item_id']]
    train_pairs = train_df[['user_id', 'item_id']]

    br_cfg = cfg['models']['bert4rec']

    dataset = BERT4RecDataset(
        train_df, n_items,
        max_len=br_cfg['max_len'], mask_prob=br_cfg['mask_prob'],
    )
    loader = DataLoader(
        dataset, batch_size=br_cfg['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True,
    )
    logger.info(f'BERT4Rec dataset: {len(dataset)} users, {len(loader)} batches')

    model = BERT4Rec(
        n_items=n_items,
        hidden=br_cfg['hidden'],
        max_len=br_cfg['max_len'],
        num_blocks=br_cfg['num_blocks'],
        num_heads=br_cfg['num_heads'],
        dropout=br_cfg['dropout'],
    ).to(device)
    logger.info(f'BERT4Rec params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=br_cfg['lr'], weight_decay=br_cfg['weight_decay'],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    user_seqs = build_user_sequences_bert(train_df, n_items, max_len=br_cfg['max_len'])

    wandb.init(
        project=cfg['wandb']['project'],
        group='bert4rec',
        name='bert4rec_cloze',
        config={**br_cfg, 'loss': 'CE_cloze', 'n_items': n_items},
    )

    best_hr10 = 0.0
    patience_count = 0

    for epoch in range(1, br_cfg['epochs'] + 1):
        model.train()
        total_loss = 0.0
        for seqs, labels in loader:
            seqs, labels = seqs.to(device), labels.to(device)
            out = model.forward(seqs)                          # (B, L, hidden)
            item_w = model.item_emb.weight[1:n_items + 1]      # (n_items, hidden)
            logits = out @ item_w.T                            # (B, L, n_items)
            loss = criterion(logits.view(-1, n_items), labels.view(-1))
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
            torch.save(model.state_dict(), models_dir / 'bert4rec_best.pt')
        else:
            patience_count += 1
            if patience_count >= br_cfg['patience']:
                logger.info(f'Early stopping at epoch {epoch}')
                wandb.log(log_dict)
                break

        wandb.log(log_dict)

    wandb.log({
        'best_HR@10': best_hr10,
        'sasrec_HR@10': SASREC_HR10,
        'svd_baseline_HR@10': SVD_BASELINE_HR10,
    })
    wandb.finish()
    logger.info(f'Section 7 BERT4Rec complete. Best HR@10={best_hr10:.4f}')


if __name__ == '__main__':
    run()
