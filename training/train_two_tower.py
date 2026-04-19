import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import wandb
import time
from config.config_loader import get_config, get_wandb_config
from utils.logger import get_logger
from utils.seed import set_seed
from utils.device import get_device, log_device_info
from models.two_tower import TwoTowerModel
from inference.retrieval import FAISSRetriever
from evaluation.metrics import evaluate_all


logger = get_logger('train_two_tower')
SVD_BASELINE_HIT10 = 0.011  # factors=64 val hit@10


class InteractionDataset(Dataset):
    """
    Each sample: (user_history_emb, target_item_emb, price_bucket, popularity_rank)
    user_history_emb = mean of all training items the user interacted with EXCEPT target
    target = the next item (leave-one-out: last item per user held as target)
    """
    def __init__(self, interactions_df, item_embs, item_meta_df, max_seq_len=50):
        self.max_seq_len = max_seq_len
        self.item_embs = item_embs
        self.price_buckets = item_meta_df.set_index('item_id')['price_bucket'].to_dict()
        self.popularity = item_meta_df.set_index('item_id')['popularity_rank'].to_dict()
        self.samples = self._build_samples(interactions_df)
        logger.info(f'Dataset built: {len(self.samples)} samples')

    def _build_samples(self, df):
        samples = []
        grouped = df.sort_values('timestamp').groupby('user_id')['item_id'].apply(list)
        for user_id, items in grouped.items():
            if len(items) < 2:
                continue
            target = items[-1]
            history = items[:-1][-self.max_seq_len:]
            samples.append((history, target))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history, target = self.samples[idx]
        hist_embs = self.item_embs[history]
        user_emb = hist_embs.mean(axis=0).astype(np.float32)
        target_emb = self.item_embs[target].astype(np.float32)
        price = int(self.price_buckets.get(target, -1))
        pop = float(self.popularity.get(target, 0.0))
        return (
            torch.tensor(user_emb),
            torch.tensor(target_emb),
            torch.tensor(price, dtype=torch.long),
            torch.tensor(pop, dtype=torch.float32),
        )


def build_item_embeddings(model, item_embs_np, item_meta_df, device, batch_size=2048):
    """Encode all items through item tower for FAISS index."""
    logger.info(f'Encoding {len(item_embs_np)} items through item tower...')
    price_buckets = item_meta_df['price_bucket'].values
    popularity = item_meta_df['popularity_rank'].values
    all_vecs = []
    for i in range(0, len(item_embs_np), batch_size):
        batch_emb = torch.tensor(item_embs_np[i:i+batch_size]).to(device)
        batch_price = torch.tensor(price_buckets[i:i+batch_size], dtype=torch.long).to(device)
        batch_pop = torch.tensor(popularity[i:i+batch_size], dtype=torch.float32).to(device)
        vecs = model.encode_items(batch_emb, batch_price, batch_pop)
        all_vecs.append(vecs.cpu().numpy())
    return np.concatenate(all_vecs, axis=0).astype(np.float32)


def evaluate(model, interactions_df, item_embs_np, item_meta_df, cohorts, retriever, cfg, device):
    k_values = cfg['evaluation']['k_values']
    primary_k = cfg['evaluation']['primary_k']
    n_test = cfg['evaluation']['n_test_users']
    max_seq = cfg['data']['max_sequence_length']

    test_users = interactions_df['user_id'].unique()
    if len(test_users) > n_test:
        test_users = np.random.choice(test_users, n_test, replace=False)

    ground_truth = (
        interactions_df.sort_values('timestamp')
        .groupby('user_id')['item_id'].last().to_dict()
    )
    grouped = interactions_df.sort_values('timestamp').groupby('user_id')['item_id'].apply(list).to_dict()

    n_items_total = len(item_embs_np)

    query_vecs = []
    valid_users = []
    for uid in test_users:
        if uid not in ground_truth:
            continue
        items = grouped.get(uid, [])[:-1][-max_seq:]
        # Drop OOV item ids that exceed precomputed embedding range
        items = [i for i in items if i < n_items_total]
        if not items:
            user_emb = np.zeros(384, dtype=np.float32)
        else:
            user_emb = item_embs_np[items].mean(axis=0).astype(np.float32)
        query_vecs.append(user_emb)
        valid_users.append(uid)

    if not query_vecs:
        return {f'hit@{k}': 0.0 for k in k_values}

    query_np = np.stack(query_vecs)
    query_tensor = torch.tensor(query_np).to(device)
    encoded = model.encode_user(query_tensor).cpu().numpy().astype(np.float32)
    _, indices = retriever.search(encoded, k=max(k_values))

    recommendations = {uid: list(indices[i]) for i, uid in enumerate(valid_users)}
    popularity_scores = item_meta_df.set_index('item_id')['popularity_rank'].reindex(
        range(n_items_total), fill_value=0).values

    metrics = evaluate_all(
        ground_truth={u: ground_truth[u] for u in valid_users if u in ground_truth},
        recommendations=recommendations,
        k_values=k_values,
        n_total_items=n_items_total,
        popularity_scores=popularity_scores,
    )

    for cohort_name in ['cold', 'sparse_low', 'sparse_high', 'warm']:
        cohort_users = set(cohorts[cohorts['cohort'] == cohort_name]['user_id'])
        cohort_recs = {u: r for u, r in recommendations.items() if u in cohort_users}
        cohort_gt = {u: ground_truth[u] for u in cohort_recs if u in ground_truth}
        if not cohort_gt:
            metrics[f'hit@{primary_k}_{cohort_name}'] = 0.0
            continue
        hits = [1.0 if cohort_gt[u] in cohort_recs[u][:primary_k] else 0.0 for u in cohort_gt]
        metrics[f'hit@{primary_k}_{cohort_name}'] = float(np.mean(hits))
    return metrics


def train():
    set_seed()
    cfg = get_config()
    wandb_cfg = get_wandb_config()
    device = get_device()
    log_device_info()

    processed = Path(cfg['data']['processed_dir'])
    tt_cfg = cfg['models']['two_tower']
    primary_k = cfg['evaluation']['primary_k']
    max_seq = cfg['data']['max_sequence_length']

    logger.info('Loading data...')
    train_df = pd.read_parquet(processed / 'train.parquet')
    val_df = pd.read_parquet(processed / 'val.parquet')
    test_df = pd.read_parquet(processed / 'test.parquet')
    cohorts = pd.read_parquet(processed / 'cohorts' / 'user_cohorts.parquet')
    item_meta = pd.read_parquet(processed / 'item_metadata.parquet')
    item_embs = np.load(Path(tt_cfg['embeddings_path']) / 'item_title_embeddings.npy')
    logger.info(f'Item embeddings: {item_embs.shape}')

    dataset = InteractionDataset(train_df, item_embs, item_meta, max_seq)
    loader = DataLoader(
        dataset, batch_size=tt_cfg['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    model = TwoTowerModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model parameters: {n_params:,}')

    optimizer = AdamW(model.parameters(), lr=tt_cfg['learning_rate'], weight_decay=tt_cfg['weight_decay'])
    total_steps = len(loader) * tt_cfg['max_epochs']
    scheduler = OneCycleLR(
        optimizer, max_lr=tt_cfg['learning_rate'],
        total_steps=total_steps,
        pct_start=tt_cfg['warmup_steps'] / max(total_steps, 1),
    )

    wandb.init(
        project=wandb_cfg['project'],
        entity=wandb_cfg['entity'],
        name='two_tower_content',
        tags=cfg['wandb']['tags'] + ['two_tower', 'infonce'],
        config={
            'model': 'two_tower',
            'user_tower': 'mean_pooling',
            'item_tower': 'st_emb+price+popularity',
            'loss': 'infonce',
            'batch_size': tt_cfg['batch_size'],
            'lr': tt_cfg['learning_rate'],
            'temperature': tt_cfg['temperature'],
            'output_dim': tt_cfg['item_embedding_dim'],
            'n_params': n_params,
            'svd_baseline_hit10': SVD_BASELINE_HIT10,
            'dataset': cfg['data']['category'],
            'seed': cfg['project']['seed'],
        },
        reinit=True,
    )
    wandb.watch(model, log='gradients', log_freq=100)

    best_hit = -1.0
    best_epoch = -1
    patience_counter = 0
    patience = tt_cfg['early_stopping_patience']
    save_path = Path(tt_cfg['save_path'])
    save_path.mkdir(parents=True, exist_ok=True)
    retriever = None

    for epoch in range(tt_cfg['max_epochs']):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for batch_idx, (user_emb, item_emb, price, pop) in enumerate(loader):
            user_emb = user_emb.to(device, non_blocking=True)
            item_emb = item_emb.to(device, non_blocking=True)
            price = price.to(device, non_blocking=True)
            pop = pop.to(device, non_blocking=True)

            optimizer.zero_grad()
            user_vec, item_vec = model(user_emb, item_emb, price, pop)
            loss = model.infonce_loss(user_vec, item_vec)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

            if batch_idx % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': scheduler.get_last_lr()[0],
                    'train/gpu_memory_gb': torch.cuda.memory_allocated(0) / 1e9 if device.type == 'cuda' else 0,
                    'train/gpu_memory_reserved_gb': torch.cuda.memory_reserved(0) / 1e9 if device.type == 'cuda' else 0,
                })

        epoch_time = time.time() - t0
        avg_loss = float(np.mean(epoch_losses))
        logger.info(f'Epoch {epoch+1}: loss={avg_loss:.4f}, time={epoch_time:.1f}s')

        item_vecs = build_item_embeddings(model, item_embs, item_meta, device)
        retriever = FAISSRetriever()
        retriever.build(item_vecs)

        val_metrics = evaluate(model, val_df, item_embs, item_meta, cohorts, retriever, cfg, device)
        wandb.log({
            **{f'val/{k}': v for k, v in val_metrics.items()},
            'epoch': epoch + 1,
            'train/epoch_loss': avg_loss,
            'train/epoch_time_s': epoch_time,
            'val/svd_baseline_hit10': SVD_BASELINE_HIT10,
        })
        logger.info(f'Val metrics: {val_metrics}')

        current_hit = val_metrics[f'hit@{primary_k}']
        if current_hit > best_hit:
            best_hit = current_hit
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), save_path / 'best_model.pt')
            retriever.save()
            logger.info(f'New best: hit@{primary_k}={best_hit:.4f} at epoch {best_epoch}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f'Early stopping at epoch {epoch+1}')
                break

    # Load best model + index for test eval
    model.load_state_dict(torch.load(save_path / 'best_model.pt', map_location=device))
    if retriever is None:
        retriever = FAISSRetriever()
    retriever.load()
    test_metrics = evaluate(model, test_df, item_embs, item_meta, cohorts, retriever, cfg, device)
    wandb.log({
        **{f'test/{k}': v for k, v in test_metrics.items()},
        'test/svd_baseline_hit10': SVD_BASELINE_HIT10,
        'test/improvement_over_svd_pct': (test_metrics[f'hit@{primary_k}'] - SVD_BASELINE_HIT10) / SVD_BASELINE_HIT10 * 100 if SVD_BASELINE_HIT10 > 0 else 0,
        'best_epoch': best_epoch,
    })
    logger.info(f'Test metrics: {test_metrics}')

    # Incremental update demo: warm-start fine-tune on val data
    logger.info('Incremental update demo: warm-start fine-tune on val data...')
    finetune_dataset = InteractionDataset(val_df, item_embs, item_meta, max_seq)
    finetune_loader = DataLoader(
        finetune_dataset, batch_size=tt_cfg['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    ft_optimizer = AdamW(model.parameters(), lr=tt_cfg['learning_rate'] * 0.1)
    model.train()
    ft_losses = []
    t0 = time.time()
    for user_emb, item_emb, price, pop in finetune_loader:
        user_emb, item_emb = user_emb.to(device), item_emb.to(device)
        price, pop = price.to(device), pop.to(device)
        ft_optimizer.zero_grad()
        user_vec, item_vec = model(user_emb, item_emb, price, pop)
        loss = model.infonce_loss(user_vec, item_vec)
        loss.backward()
        ft_optimizer.step()
        ft_losses.append(loss.item())
    ft_time = time.time() - t0
    wandb.log({
        'incremental/finetune_loss': float(np.mean(ft_losses)) if ft_losses else 0.0,
        'incremental/finetune_time_s': ft_time,
        'incremental/update_type': 'warm_start',
    })
    logger.info(f'Warm-start fine-tune complete: loss={np.mean(ft_losses):.4f}, time={ft_time:.1f}s')

    wandb.finish()
    logger.info(f'Two-Tower training complete. Best val hit@{primary_k}={best_hit:.4f}')
    return model, test_metrics


if __name__ == '__main__':
    train()
