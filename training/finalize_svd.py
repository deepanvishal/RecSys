"""
Finalize Section 4 after factors=256 was killed mid-training.
- Picks factors=64 as best (val/hit@10=1.1% beats both CF baseline and factors=128).
- Runs test eval on best model.
- Saves best to artifacts/models/svd/best/.
- Logs to W&B: test eval run + a documentation run for the killed factors=256.
"""
from pathlib import Path
import wandb
from config.config_loader import get_config, get_wandb_config
from utils.logger import get_logger
from utils.seed import set_seed
from models.svd import SVDModel
from training.train_svd import load_data, evaluate, CF_BASELINE_HIT10


logger = get_logger('finalize_svd')


# Validation results from the actual training runs (24-thread runs in /tmp/train_svd.log)
VAL_HIT10 = {64: 0.011, 128: 0.006}
BEST_FACTORS = max(VAL_HIT10, key=VAL_HIT10.get)
BEST_VAL_HIT10 = VAL_HIT10[BEST_FACTORS]


def main():
    set_seed()
    cfg = get_config()
    wandb_cfg = get_wandb_config()
    save_path = cfg['models']['svd']['save_path']
    primary_k = cfg['evaluation']['primary_k']

    logger.info(f'Loading best SVD: factors={BEST_FACTORS} (val/hit@{primary_k}={BEST_VAL_HIT10})')
    best_model = SVDModel.load(Path(save_path) / f'svd_factors{BEST_FACTORS}.pkl')
    logger.info('Loaded.')

    _, _, _, test, cohorts = load_data(cfg)

    logger.info('Running test eval (use_foldin=True for sparse users)...')
    test_metrics = evaluate(best_model, test, cohorts, cfg, use_foldin=True)
    logger.info(f'Test metrics: {test_metrics}')

    # W&B run 1: test eval for best model
    wandb.init(
        project=wandb_cfg['project'],
        entity=wandb_cfg['entity'],
        name=f'svd_factors{BEST_FACTORS}_test_eval',
        tags=cfg['wandb']['tags'] + ['svd', 'test', 'best'],
        config={
            'model': 'svd',
            'factors': BEST_FACTORS,
            'note': 'Best of completed runs. factors=256 killed due to WSL memory pressure.',
        },
        reinit=True,
    )
    wandb.log({f'test/{k}': v for k, v in test_metrics.items()})
    wandb.log({
        'test/cf_baseline_hit10': CF_BASELINE_HIT10,
        'test/improvement_over_cf_pct': (test_metrics[f'hit@{primary_k}'] - CF_BASELINE_HIT10) / CF_BASELINE_HIT10 * 100,
    })
    wandb.finish()

    # W&B run 2: documentation of skipped factors=256
    wandb.init(
        project=wandb_cfg['project'],
        entity=wandb_cfg['entity'],
        name='svd_factors256_skipped',
        tags=cfg['wandb']['tags'] + ['svd', 'skipped', 'limitation'],
        config={
            'model': 'svd',
            'factors': 256,
            'status': 'killed',
            'reason': 'WSL allocates ~30GB of host RAM by default; factors=256 ALS exceeded this and began swapping (~10 min/iter vs ~30 sec/iter expected). Killed at iter 4/20.',
            'observed_per_iter_seconds_in_swap': 590,
            'projected_total_minutes_if_completed': 162,
            'mitigation_options': 'increase WSL memory cap via .wslconfig OR run on a Linux host with native RAM access OR keep factors<=128 in this environment',
        },
        reinit=True,
    )
    wandb.log({'limitation': 1.0})
    wandb.finish()

    # Save best
    best_model.save(save_path + 'best/')
    logger.info(f'Best model saved to {save_path}best/')
    logger.info('Section 4 finalized.')


if __name__ == '__main__':
    main()
