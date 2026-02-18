import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

file = 'cox_regression1.csv'

df = pd.read_csv(file)

df.rename(columns={
    'Status (1 = dead, 0 = censored)': 'event',
    'Survival (days)': 'duration',
    'Predicted Tumor State': 'tumor_state'
}, inplace=True)

for col in ['tumor_state', 'Age', 'event', 'duration']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['tumor_state', 'Age', 'event', 'duration'])
df = df[['tumor_state', 'Age', 'duration', 'event']].copy()

print(f"samples: {len(df)}, events: {int(df['event'].sum())}, censored: {len(df) - int(df['event'].sum())}\n")

cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='event', show_progress=False)

s = cph.summary
results = pd.DataFrame({
    'covariate':         s.index,
    'coef':              s['coef'].values,
    'HR':                np.exp(s['coef']).values,
    'se':                s['se(coef)'].values,
    'CI_low':            np.exp(s['coef lower 95%']).values,
    'CI_high':           np.exp(s['coef upper 95%']).values,
    'z':                 (s['coef'] / s['se(coef)']).values,
    'p':                 s['p'].values,
    '-log2p':            -np.log2(s['p']).values
})

print(results.to_string(index=False))
print()

ll_model = cph.log_likelihood_
ll_null   = cph.log_likelihood_null_
lrt       = -2 * (ll_null - ll_model)
lrt_p     = 1 - chi2.cdf(lrt, df=len(s))

print(f"n={len(df)}  concordance={cph.concordance_index_:.4f}  AIC={cph.AIC_partial_:.2f}")
print(f"LR test: stat={lrt:.4f}  df={len(s)}  p={lrt_p:.2e}  -log2p={-np.log2(lrt_p):.2f}")
print()

for _, row in results.iterrows():
    direction = f"increases hazard by {(row['HR']-1)*100:.1f}%" if row['HR'] > 1 else f"decreases hazard by {(1-row['HR'])*100:.1f}%"
    sig = "  *" if row['p'] < 0.05 else ""
    print(f"{row['covariate']}: HR={row['HR']:.4f}, 95%CI ({row['CI_low']:.4f}-{row['CI_high']:.4f}), p={row['p']:.4f}{sig}")
    print(f"  -> per unit increase {direction}")

results.to_csv('cox_results.csv', index=False)
