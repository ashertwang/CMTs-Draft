# -*- coding: utf-8 -*-
"""mvb.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/10OeoU3rt-kSzL2CXjjgJ4wZNscg6U02Q
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import ADASYN

data = pd.read_csv("../Data/mvb_unadjmetaboliteprofiling.csv")

features = data.columns[2:]
target_column = 'Tumor State'

if data[target_column].isnull().any():
    raise ValueError("NaN values found in the target variable.")

X = data[features]
y = data[target_column]

imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

data[target_column] = data[target_column].map({'Tumor': 1, 'Healthy': 0})

adasyn = ADASYN(random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_imputed, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_dist = {'n_estimators': [100],
              'max_depth': [30],
              'min_samples_split': [5],
              'min_samples_leaf': [1]}

rf_classifier = RandomForestClassifier(random_state=42)

random_search = RandomizedSearchCV(rf_classifier, param_distributions=param_dist, n_iter=20, cv=3, scoring='accuracy', random_state=42)

random_search.fit(X_train_scaled, y_train)

best_rf = random_search.best_estimator_

y_pred = best_rf.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Best Parameters: {random_search.best_params_}')
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{confusion_mat}')
print(f'Classification Report:\n{classification_rep}')

# Feature Importance
sorted_features = X_imputed.columns[best_rf.feature_importances_.argsort()[::-1]]
sorted_importances = best_rf.feature_importances_[best_rf.feature_importances_.argsort()[::-1]]

print("Feature Importance:")
for feature, importance in zip(sorted_features, sorted_importances):
    print(f"{feature}: {importance}")

# Permutation Importance
perm_importance = permutation_importance(best_rf, X_test_scaled, y_test, n_repeats=30, random_state=42)
perm_sorted_features = X_imputed.columns[perm_importance.importances_mean.argsort()[::-1]]
perm_sorted_importances = perm_importance.importances_mean[perm_importance.importances_mean.argsort()[::-1]]

print("\nPermutation Importance:")
for feature, importance in zip(perm_sorted_features, perm_sorted_importances):
    print(f"{feature}: {importance}")
