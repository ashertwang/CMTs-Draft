import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import ADASYN
import warnings
warnings.filterwarnings('ignore')

raw = pd.read_csv("Mammary_Data_Plus_Age.csv")
raw = raw.dropna(subset=['Tumor State', 'Age[Month]'])

stuff = [c for c in raw.columns if c not in ['Source Name', 'Tumor State', 'Age[Month]']]

filler = SimpleImputer(strategy='mean')
raw[stuff] = filler.fit_transform(raw[stuff])

raw['bucket'] = pd.qcut(raw['Age[Month]'], q=4, labels=[1,2,3,4]).astype(int)

r, p = stats.pointbiserialr(raw['Age[Month]'], raw['Tumor State'])
print(f"samples: {len(raw)}  malignant: {int(raw['Tumor State'].sum())}  benign: {int((raw['Tumor State']==0).sum())}")
print(f"age vs tumor: r={r:.4f}  p={p:.4f}\n")

rows = []
for met in stuff:
    X = raw[['Age[Month]', 'Tumor State']].values
    y = raw[met].values
    fit = LinearRegression().fit(X, y)
    n, k = len(y), X.shape[1]
    resid = y - fit.predict(X)
    mse = np.sum(resid**2) / (n - k - 1)
    Xb = np.column_stack([np.ones(n), X])
    se = np.sqrt(mse * np.linalg.pinv(Xb.T @ Xb).diagonal()[1:])
    t = fit.coef_ / se
    pv = [2 * (1 - stats.t.cdf(abs(ti), df=n-k-1)) for ti in t]
    rows.append({'met': met, 'age_b': fit.coef_[0], 'age_p': pv[0], 'tumor_b': fit.coef_[1], 'tumor_p': pv[1], 'r2': fit.score(X, y)})

ols = pd.DataFrame(rows)
print(f"OLS sig tumor: {(ols['tumor_p']<0.05).sum()}  sig age: {(ols['age_p']<0.05).sum()}")
print(ols.sort_values('tumor_p')[['met','tumor_b','tumor_p','age_b','age_p']].head(5).to_string(index=False))
print()
print(ols.sort_values('age_p')[['met','age_b','age_p','tumor_b','tumor_p']].head(5).to_string(index=False))

mw_rows = []
for g in [1,2,3,4]:
    chunk = raw[raw['bucket']==g]
    if chunk['Tumor State'].nunique() < 2:
        continue
    for met in stuff:
        neg = chunk[chunk['Tumor State']==0][met]
        pos = chunk[chunk['Tumor State']==1][met]
        if len(neg) < 2 or len(pos) < 2:
            continue
        u, pv = stats.mannwhitneyu(neg, pos)
        mw_rows.append({'met': met, 'bucket': g, 'p': pv})

mw = pd.DataFrame(mw_rows).sort_values('p')
print(f"\nMann-Whitney sig within buckets: {(mw['p']<0.05).sum()}")
print(mw.head(5).to_string(index=False))

kw_rows = []
for met in stuff:
    gs = [raw[raw['bucket']==g][met].values for g in [1,2,3,4]]
    stat, pv = stats.kruskal(*gs)
    kw_rows.append({'met': met, 'stat': stat, 'p': pv})

kw = pd.DataFrame(kw_rows).sort_values('p')
print(f"\nKruskal-Wallis sig across buckets: {(kw['p']<0.05).sum()}")
print(kw.head(5).to_string(index=False))

all_features = raw[['Age[Month]'] + stuff]
labels = raw['Tumor State']

scaler = StandardScaler()
X_bal, y_bal = ADASYN(random_state=42).fit_resample(all_features, labels)
X_bal = pd.DataFrame(scaler.fit_transform(X_bal), columns=all_features.columns)

X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
hunt = RandomizedSearchCV(RandomForestClassifier(random_state=42), grid, n_iter=20, cv=3, scoring='accuracy', random_state=42)
hunt.fit(X_train, y_train)

winner = hunt.best_estimator_
guesses = winner.predict(X_test)

print(f"\nRF accuracy: {accuracy_score(y_test, guesses):.4f}")
print(f"best params: {hunt.best_params_}")
print(classification_report(y_test, guesses, target_names=['benign','malignant']))

ranked = pd.DataFrame({'feature': all_features.columns, 'importance': winner.feature_importances_}).sort_values('importance', ascending=False)
age_spot = ranked['feature'].tolist().index('Age[Month]') + 1
print(f"Age[Month] rank: {age_spot} / {len(ranked)}")
print(ranked.head(10).to_string(index=False))

ols.sort_values('tumor_p').to_csv('ols_results.csv', index=False)
kw.to_csv('kw_results.csv', index=False)
mw.to_csv('mw_results.csv', index=False)
ranked.to_csv('rf_importance.csv', index=False)
