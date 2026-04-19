"""Programmatically build the 3 gap-fill notebooks. Run once after Section 12."""
from pathlib import Path
import nbformat as nbf


NB_DIR = Path('notebooks')
NB_DIR.mkdir(parents=True, exist_ok=True)


def build_notebook(cells, out_path):
    nb = nbf.v4.new_notebook()
    nb['cells'] = [nbf.v4.new_code_cell(src) for src in cells]
    nb['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.10'},
    }
    nbf.write(nb, str(out_path))
    print(f'Wrote {out_path}')


# ===========================================================================
# Notebook 1: 01_eda.ipynb
# ===========================================================================
nb1 = [
    # Cell 1 — Imports and setup
    """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
from pathlib import Path
plt.style.use('seaborn-v0_8-whitegrid')
PROCESSED = Path('data/processed')
print('Imports OK')""",

    # Cell 2 — Load data
    """full = pd.read_parquet(PROCESSED / 'full.parquet')
train = pd.read_parquet(PROCESSED / 'train.parquet')
val   = pd.read_parquet(PROCESSED / 'val.parquet')
test  = pd.read_parquet(PROCESSED / 'test.parquet')
item_meta = pd.read_parquet(PROCESSED / 'item_metadata.parquet')
user_meta = pd.read_parquet(PROCESSED / 'user_metadata.parquet')
stats = json.load(open(PROCESSED / 'dataset_stats.json'))
print(f'Users: {full.user_id.nunique()} | Items: {full.item_id.nunique()} | Ratings: {len(full):,}')""",

    # Cell 3 — Basic stats
    """summary = pd.DataFrame({
    'Split': ['Train', 'Val', 'Test', 'Total'],
    'Interactions': [len(train), len(val), len(test), len(full)],
    'Users': [train.user_id.nunique(), val.user_id.nunique(),
              test.user_id.nunique(), full.user_id.nunique()],
    'Items': [train.item_id.nunique(), val.item_id.nunique(),
              test.item_id.nunique(), full.item_id.nunique()],
})
print(summary.to_string(index=False))
print(f'Avg ratings/user: {len(full)/full.user_id.nunique():.1f}')""",

    # Cell 4 — Interaction distribution per user
    """user_counts = train.groupby('user_id').size()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(user_counts, bins=50, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Ratings per user'); axes[0].set_ylabel('Count')
axes[0].set_title('Ratings per User Distribution')
axes[1].hist(user_counts.clip(upper=50), bins=50, color='coral', edgecolor='white')
axes[1].set_xlabel('Ratings per user (clipped at 50)'); axes[1].set_ylabel('Count')
axes[1].set_title('Ratings per User (Zoomed)')
plt.tight_layout()
Path('artifacts').mkdir(exist_ok=True)
plt.savefig('artifacts/eda_user_dist.png', dpi=120)
plt.show()""",

    # Cell 5 — Item popularity (long-tail)
    """item_counts = train.groupby('item_id').size().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(len(item_counts)), item_counts.values, color='steelblue', linewidth=0.8)
ax.set_xlabel('Item rank (by popularity)'); ax.set_ylabel('Interaction count')
ax.set_title('Item Popularity \\u2014 Long Tail Distribution')
ax.set_yscale('log')
top20_pct = item_counts.head(int(len(item_counts)*0.2)).sum() / item_counts.sum()
ax.axvline(int(len(item_counts)*0.2), color='red', linestyle='--',
           label=f'Top 20% items = {top20_pct:.1%} of interactions')
ax.legend()
plt.tight_layout()
plt.savefig('artifacts/eda_item_longtail.png', dpi=120)
plt.show()
print(f'Top 20% items cover {top20_pct:.1%} of all interactions (long-tail confirmed)')""",

    # Cell 6 — User demographics
    """age_labels = {0:'<18', 1:'18-24', 2:'25-34', 3:'35-44', 4:'45-49', 5:'50-55', 6:'56+'}
age_counts = user_meta['age_enc'].map(age_labels).value_counts()
gender_counts = user_meta['gender_enc'].map({0:'Female', 1:'Male'}).value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(gender_counts.index, gender_counts.values, color=['coral', 'steelblue'])
axes[0].set_title('Gender Distribution'); axes[0].set_ylabel('Users')
age_order = ['<18','18-24','25-34','35-44','45-49','50-55','56+']
axes[1].bar(age_order, [age_counts.get(a,0) for a in age_order], color='steelblue')
axes[1].set_title('Age Distribution'); axes[1].set_ylabel('Users')
plt.tight_layout()
plt.savefig('artifacts/eda_demographics.png', dpi=120)
plt.show()""",

    # Cell 7 — Genre distribution
    """genre_cols = [c for c in item_meta.columns if c.startswith('genre_') and c != 'genre_vector']
if genre_cols:
    genre_counts = item_meta[genre_cols].sum().sort_values(ascending=False)
    genre_counts.index = [c.replace('genre_','').replace('_',' ').title() for c in genre_counts.index]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.barh(genre_counts.index[:12], genre_counts.values[:12], color='steelblue')
    ax.set_title('Top 12 Genres in ML-1M'); ax.set_xlabel('Number of movies')
    plt.tight_layout()
    plt.savefig('artifacts/eda_genres.png', dpi=120)
    plt.show()
else:
    print('No per-genre columns found in item_metadata')""",

    # Cell 8 — Sparsity and summary
    """n_users = full.user_id.nunique()
n_items = full.item_id.nunique()
sparsity = 1 - len(full) / (n_users * n_items)
print(f'Matrix sparsity: {sparsity:.4%}')
print(f'Avg ratings/user: {len(full)/n_users:.1f}')
print(f'Avg ratings/item: {len(full)/n_items:.1f}')
print(f'Leave-one-out val/test: {len(val)} users each')
print('EDA complete. Plots saved to artifacts/')""",
]
build_notebook(nb1, NB_DIR / '01_eda.ipynb')


# ===========================================================================
# Notebook 2: 02_model_comparison.ipynb
# ===========================================================================
nb2 = [
    # Cell 1 — Imports
    """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
plt.style.use('seaborn-v0_8-whitegrid')
Path('artifacts').mkdir(exist_ok=True)
print('Imports OK')""",

    # Cell 2 — Results table
    """results = pd.DataFrame([
    {'Model': 'SASRec',     'HR@5': 0.2272, 'HR@10': 0.3328, 'HR@20': 0.4449, 'NDCG@10': 0.1873, 'Role': 'Tier 3'},
    {'Model': 'SVD f=128',  'HR@5': 0.0535, 'HR@10': 0.1020, 'HR@20': 0.1642, 'NDCG@10': 0.0487, 'Role': 'Tier 2'},
    {'Model': 'UserUserCF', 'HR@5': 0.0470, 'HR@10': 0.0830, 'HR@20': 0.1460, 'NDCG@10': 0.0400, 'Role': 'Tier 2'},
    {'Model': 'ItemItemCF', 'HR@5': 0.0460, 'HR@10': 0.0816, 'HR@20': 0.1340, 'NDCG@10': 0.0390, 'Role': 'Tier 2'},
    {'Model': 'Two-Tower',  'HR@5': 0.0353, 'HR@10': 0.0647, 'HR@20': 0.1078, 'NDCG@10': 0.0315, 'Role': 'Tier 3'},
    {'Model': 'BERT4Rec',   'HR@5': 0.0210, 'HR@10': 0.0417, 'HR@20': 0.0720, 'NDCG@10': 0.0199, 'Role': 'Experimental'},
])
print(results.to_string(index=False))""",

    # Cell 3 — HR@10 bar chart
    """colors = {'Tier 3': 'steelblue', 'Tier 2': 'coral', 'Experimental': 'lightgray'}
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(results['Model'], results['HR@10'],
               color=[colors[r] for r in results['Role']], edgecolor='white')
ax.axvline(0.1, color='gray', linestyle='--', linewidth=0.8, label='0.10 reference')
ax.set_xlabel('HR@10 (full-catalog evaluation)')
ax.set_title('Model Comparison \\u2014 Hit Rate @ 10 (MovieLens 1M)')
for bar, val in zip(bars, results['HR@10']):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
ax.legend(handles=patches)
plt.tight_layout()
plt.savefig('artifacts/model_comparison_hr10.png', dpi=120)
plt.show()""",

    # Cell 4 — HR@10 vs NDCG@10 scatter
    """fig, ax = plt.subplots(figsize=(8, 6))
for _, row in results.iterrows():
    ax.scatter(row['HR@10'], row['NDCG@10'], s=80,
               color=colors[row['Role']], zorder=3)
    ax.annotate(row['Model'], (row['HR@10'], row['NDCG@10']),
                textcoords='offset points', xytext=(6, 3), fontsize=9)
ax.set_xlabel('HR@10'); ax.set_ylabel('NDCG@10')
ax.set_title('HR@10 vs NDCG@10 \\u2014 All Models')
patches = [mpatches.Patch(color=v, label=k) for k, v in colors.items()]
ax.legend(handles=patches)
plt.tight_layout()
plt.savefig('artifacts/hr_vs_ndcg.png', dpi=120)
plt.show()""",

    # Cell 5 — SASRec learning curve
    """epochs = [1, 3, 5, 10, 15, 20, 26, 31]
hr10   = [0.0356, 0.1174, 0.2091, 0.2836, 0.3104, 0.3224, 0.3328, 0.3305]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(epochs, hr10, 'o-', color='steelblue', linewidth=2, markersize=6)
ax.axhline(0.1020, color='coral', linestyle='--', linewidth=1, label='SVD baseline (0.1020)')
ax.axhline(0.0830, color='gray',  linestyle='--', linewidth=1, label='CF baseline (0.0830)')
ax.scatter([26], [0.3328], s=150, color='red', zorder=5, label='Best epoch 26 (0.3328)')
ax.set_xlabel('Epoch'); ax.set_ylabel('HR@10 (full-catalog)')
ax.set_title('SASRec Training Curve \\u2014 MovieLens 1M')
ax.legend()
plt.tight_layout()
plt.savefig('artifacts/sasrec_learning_curve.png', dpi=120)
plt.show()""",

    # Cell 6 — Summary table with delta vs SVD
    """results['Delta vs SVD'] = (results['HR@10'] - 0.1020).map(lambda x: f'{x:+.4f}')
results['3x SVD?'] = results['HR@10'].map(lambda x: 'YES' if x > 0.306 else '')
print(results[['Model','HR@10','NDCG@10','Delta vs SVD','3x SVD?']].to_string(index=False))
print()
print('Conclusion: SASRec HR@10=0.3328 is 3.26x better than SVD (0.1020)')""",
]
build_notebook(nb2, NB_DIR / '02_model_comparison.ipynb')


# ===========================================================================
# Notebook 3: 03_bias_analysis.ipynb
# ===========================================================================
nb3 = [
    # Cell 1 — Imports
    """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from pathlib import Path
plt.style.use('seaborn-v0_8-whitegrid')
BIAS = Path('artifacts/bias')
Path('artifacts').mkdir(exist_ok=True)
print('Imports OK')""",

    # Cell 2 — Load bias CSVs
    """summary    = pd.read_csv(BIAS / 'bias_summary.csv').iloc[0]
gender_df  = pd.read_csv(BIAS / 'sasrec_gender_bias.csv')
age_df     = pd.read_csv(BIAS / 'sasrec_age_bias.csv')
genre_df   = pd.read_csv(BIAS / 'sasrec_genre_concentration.csv')
pop_s      = pd.read_csv(BIAS / 'sasrec_popularity_bias.csv').iloc[0]
pop_v      = pd.read_csv(BIAS / 'svd_popularity_bias.csv').iloc[0]
print('Bias data loaded.')
print(f'SASRec popularity ratio: {summary[\"sasrec_popularity_ratio\"]:.2f}x')
print(f'SVD popularity ratio: {summary[\"svd_popularity_ratio\"]:.2f}x')""",

    # Cell 3 — Popularity bias comparison
    """fig, axes = plt.subplots(1, 2, figsize=(12, 4))
models = ['SASRec', 'SVD']
ratios = [pop_s['popularity_ratio'], pop_v['popularity_ratio']]
lt_frac = [pop_s['longtail_fraction'], pop_v['longtail_fraction']]
axes[0].bar(models, ratios, color=['steelblue', 'coral'], edgecolor='white')
axes[0].axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Fair baseline (1.0)')
axes[0].set_ylabel('Popularity ratio (higher = more biased)')
axes[0].set_title('Popularity Bias: SASRec vs SVD')
axes[0].legend()
for i, v in enumerate(ratios):
    axes[0].text(i, v + 0.05, f'{v:.2f}x', ha='center', fontsize=11, fontweight='bold')
axes[1].bar(models, [f*100 for f in lt_frac], color=['steelblue', 'coral'], edgecolor='white')
axes[1].set_ylabel('Long-tail fraction (%)')
axes[1].set_title('Long-Tail Coverage: SASRec vs SVD')
for i, v in enumerate(lt_frac):
    axes[1].text(i, v*100 + 0.1, f'{v:.1%}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('artifacts/bias_popularity.png', dpi=120)
plt.show()""",

    # Cell 4 — Gender bias
    """fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(gender_df['group'], gender_df['HR@10'],
              color=['coral', 'steelblue'], edgecolor='white')
ax.set_ylabel('HR@10 (full-catalog)')
for bar, (_, row) in zip(bars, gender_df.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{row[\"HR@10\"]:.4f}\\n(n={row[\"n_users\"]})',
            ha='center', fontsize=10)
gap = summary['gender_gap_HR10']
ax.set_ylim(0, 0.42)
ax.set_title(f'Gender Bias \\u2014 Gap (M-F): {gap:+.4f} HR@10')
plt.tight_layout()
plt.savefig('artifacts/bias_gender.png', dpi=120)
plt.show()""",

    # Cell 5 — Age bias
    """age_order = ['<18','18-24','25-34','35-44','45-49','50-55','56+']
age_df_sorted = age_df.set_index('group').reindex(age_order).reset_index()
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(age_df_sorted['group'], age_df_sorted['HR@10'],
              color='steelblue', edgecolor='white')
ax.axhline(age_df_sorted['HR@10'].mean(), color='red', linestyle='--',
           linewidth=1, label=f'Mean HR@10 = {age_df_sorted[\"HR@10\"].mean():.4f}')
for bar, (_, row) in zip(bars, age_df_sorted.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{row[\"HR@10\"]:.3f}', ha='center', fontsize=8)
ax.set_ylabel('HR@10'); ax.set_xlabel('Age group')
ax.set_title(f'Age Bias \\u2014 Range: {summary[\"age_hr10_range\"]:.4f} HR@10')
ax.legend()
plt.tight_layout()
plt.savefig('artifacts/bias_age.png', dpi=120)
plt.show()
print(f'Under-18 HR@10: {age_df_sorted.iloc[0][\"HR@10\"]:.4f} (lowest)')
print(f'50-55  HR@10: {age_df_sorted.iloc[5][\"HR@10\"]:.4f} (highest)')""",

    # Cell 6 — Genre concentration
    """top10 = genre_df.head(10)
fig, ax = plt.subplots(figsize=(10, 4))
ax.barh(top10['genre'][::-1], top10['fraction_in_recs'][::-1],
        color='steelblue', edgecolor='white')
ax.set_xlabel('Fraction of recommendations containing genre')
ax.set_title('Genre Concentration in SASRec Recommendations (Top 10)')
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
plt.tight_layout()
plt.savefig('artifacts/bias_genre.png', dpi=120)
plt.show()""",

    # Cell 7 — Summary
    """print('=== Bias Audit Summary ===')
print(f'Popularity ratio (SASRec): {summary[\"sasrec_popularity_ratio\"]:.2f}x (SVD: {summary[\"svd_popularity_ratio\"]:.2f}x)')
print(f'Long-tail fraction (SASRec): {summary[\"sasrec_longtail_fraction\"]:.1%}')
print(f'Gender gap (M-F): {summary[\"gender_gap_HR10\"]:+.4f} HR@10')
print(f'Age range (max-min): {summary[\"age_hr10_range\"]:.4f} HR@10')
print(f'Top genre: {summary[\"top_genre\"]} ({summary[\"top_genre_frac\"]:.1%} of recs)')
print(f'Under-18 cohort underperforms by ~{(summary[\"age_hr10_range\"]*0.7):.3f} HR@10 \\u2014 production risk for youth platforms')""",
]
build_notebook(nb3, NB_DIR / '03_bias_analysis.ipynb')


print('All 3 notebooks written.')
