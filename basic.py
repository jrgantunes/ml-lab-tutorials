#!/usr/bin/env python3

"""
Compares performance of estimators and pipelines.
"""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

IRS = [1., 3., 5., 10.]
LR = LogisticRegression(solver='lbfgs', max_iter=5000)
DT = DecisionTreeClassifier(random_state=0)
SMOTE_LR = make_pipeline(SMOTE(random_state=1), LogisticRegression(solver='lbfgs', max_iter=5000))
DT_PARAM_GRID = {'max_depth': [2, 3, 4, 5], 'criterion': ['gini', 'entropy']}
SMOTE_LR_PARAM_GRID = {'smote__k_neighbors': [2, 3, 4], 'logisticregression__C': [1e3, 1e2, 1e1, 1e0, 1e-1]}
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)


def generate_data(imbalance_ratio):
    """Generate data of given IR."""
    weights = imbalance_ratio / (imbalance_ratio + 1), 1 / (imbalance_ratio + 1)
    X, y = make_classification(n_samples=200, n_features=5, random_state=3, weights=weights)
    return X, y


def mean_cv_score(estimator, X, y):
    """Return stratified 5-fold mean f-score."""
    return cross_val_score(estimator, X, y, cv=CV, scoring='f1').mean()


def optimal_mean_cv_score(estimator, param_grid, X, y):
    """Return highest stratified 5-fold mean f-score for a parameter grid."""
    gscv = GridSearchCV(estimator, param_grid, cv=CV, scoring='f1', iid=False).fit(X, y)
    return gscv.best_score_


if __name__ =='__main__':

    # Scores placeholder
    scores = []

    # Iterate through imbalance ratios
    for imbalance_ratio in IRS:

        # Generate data
        X, y = generate_data(imbalance_ratio)
        
        # Calculate scores
        ir_scores = [
            imbalance_ratio,
            mean_cv_score(LR, X, y), 
            mean_cv_score(DT, X, y), 
            mean_cv_score(SMOTE_LR, X, y), 
            optimal_mean_cv_score(DT, DT_PARAM_GRID, X, y), 
            optimal_mean_cv_score(SMOTE_LR, SMOTE_LR_PARAM_GRID, X, y)
        ]
        scores.append(ir_scores)

    # Create and print scores table
    scores = pd.DataFrame(scores, columns=['IR', 'LR', 'DT', 'SMOTE + LR', 'Optimal DT', 'Optimal SMOTE + LR'])
    print(scores)
