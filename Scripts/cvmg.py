import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, make_scorer, confusion_matrix
warnings.filterwarnings('ignore')
raw = pd.read_csv('/mnt/user-data/uploads/mg_cradjmetaboliteprofiling.csv')

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

pipes = {
    'Logistic Regression': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('sel', SelectKBest(f_classif, k=10)),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    'SVM': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('sel', SelectKBest(f_classif, k=15)),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42))]),
    'Random Forest': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sel', SelectKBest(f_classif, k=20)),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split=3, random_state=42))]),
    'Gradient Boosting': Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scl', StandardScaler()),
        ('sel', SelectKBest(f_classif, k=15)),
        ('clf', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42))])}

grids = {
    'Logistic Regression': {'sel__k': [5, 10, 15], 'clf__C': [0.1, 1, 10], 'clf__solver': ['lbfgs']},
    'SVM':                 {'sel__k': [10, 15, 20], 'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto']},
    'Random Forest':       {'sel__k': [10, 15, 20], 'clf__max_depth': [5, 8, 10], 'clf__min_samples_split': [2, 3, 4]},
    'Gradient Boosting':   {'sel__k': [10, 15, 20], 'clf__max_depth': [3, 4, 5], 'clf__learning_rate': [0.05, 0.1]}}
scoring = {
    'auc':       'roc_auc',
    'precision': make_scorer(precision_score),
    'recall':    make_scorer(recall_score),
    'f1':        make_scorer(f1_score)}

def run_comparison(grade_a, grade_b):
    df = raw[raw['Tumor Grade'].isin([grade_a, grade_b])].copy()
    X = df.drop(['Sample', 'Tumor Grade'], axis=1)
    y = (df['Tumor Grade'] == grade_b).astype(int)
    results = {}
    for name in pipes:
        gs = GridSearchCV(pipes[name], grids[name], cv=inner_cv, scoring='roc_auc', n_jobs=-1)
        cv = cross_validate(gs, X, y, cv=outer_cv, scoring=scoring, n_jobs=-1, return_estimator=True)
        results[name] = {m: cv[f'test_{m.lower()}'] for m in ['auc', 'precision', 'recall', 'f1']}
    print(f"\nGrade {grade_a} vs Grade {grade_b}")
    header = f"{'Model':<22} {'AUC':>14} {'Precision':>14} {'Recall':>14} {'F1':>14}"
    print(header)
    for name, res in results.items():
        row = f"{name:<22}"
        for m in ['auc', 'precision', 'recall', 'f1']:
            row += f"  {res[m].mean():.3f}±{res[m].std():.3f}"
        print(row)
    best = max(results, key=lambda n: results[n]['auc'].mean())
    print(f"Best model: {best} (AUC {results[best]['auc'].mean():.3f})")
    best_gs = GridSearchCV(pipes[best], grids[best], cv=inner_cv, scoring='roc_auc', n_jobs=-1)
    preds = cross_val_predict(best_gs, X, y, cv=outer_cv, method='predict_proba')[:, 1]
    y_hat = (preds >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    ppv  = tp / (tp + fp) if (tp + fp) else 0
    npv  = tn / (tn + fn) if (tn + fn) else 0
    print(f"Sensitivity: {sens:.3f}  Specificity: {spec:.3f}  PPV: {ppv:.3f}  NPV: {npv:.3f}")
    print(f"Confusion matrix (TN={tn} FP={fp} FN={fn} TP={tp})")

run_comparison(1, 2)
run_comparison(1, 3)
run_comparison(2, 3)
