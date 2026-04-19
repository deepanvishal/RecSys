import pandas as pd
from pathlib import Path


out = Path('artifacts/bias')
required = [
    'sasrec_popularity_bias.csv',
    'sasrec_gender_bias.csv',
    'sasrec_age_bias.csv',
    'sasrec_occupation_bias.csv',
    'sasrec_genre_concentration.csv',
    'svd_popularity_bias.csv',
    'bias_summary.csv',
]
for f in required:
    assert (out / f).exists(), f'Missing {f}'
    print(f'OK: {f}')


summary = pd.read_csv(out / 'bias_summary.csv').iloc[0]
print('\nBias Summary:')
print(f'  SASRec popularity ratio:  {summary["sasrec_popularity_ratio"]:.3f} (1.0 = fair, >1 = popular-biased)')
print(f'  SASRec long-tail fraction:{summary["sasrec_longtail_fraction"]:.3f} (higher = more diverse)')
print(f'  SVD popularity ratio:     {summary["svd_popularity_ratio"]:.3f}')
print(f'  Gender gap (M-F HR@10):   {summary["gender_gap_HR10"]:+.4f}')
print(f'  Age HR@10 range:          {summary["age_hr10_range"]:.4f} (max-min across age groups)')
print(f'  Top genre in recs:        {summary["top_genre"]} ({summary["top_genre_frac"]:.1%})')


print('\nGender breakdown:')
print(pd.read_csv(out / 'sasrec_gender_bias.csv').to_string(index=False))
print('\nAge breakdown:')
print(pd.read_csv(out / 'sasrec_age_bias.csv').to_string(index=False))
print('\nTop 5 genres in recommendations:')
print(pd.read_csv(out / 'sasrec_genre_concentration.csv').head().to_string(index=False))
print('\nSection 9 bias audit complete.')
