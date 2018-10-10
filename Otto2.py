# -*- coding: utf-8 -*-
"""
Beating the benchmark 
Otto Group product classification challenge @ Kaggle
@author: Fernandroid
"""
import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation
from sklearn.externals import joblib
from sklearn.metrics import log_loss
from scipy.stats import randint ,uniform

X_train=joblib.load('...\Otto\X_train.pkl')
y_train=joblib.load('...\Otto\y_train.pkl')
X_test=joblib.load('...\Otto\X_test.pkl')
y_test=joblib.load('...\Otto\y_test.pkl')

boo = ensemble.GradientBoostingClassifier()
# specify parameters and distributions to sample from
param_dist = {"n_estimators": randint(130, 500),
              "max_depth": [5,7,8],
              "max_features": uniform(),
              "subsample": [0.6,0.7,0.8,0.9,1.],
              "learning_rate": [0.01,0.02,0.03,0.04,0.05]}

# run randomized search
n_iter_search = 3
random_search = grid_search.RandomizedSearchCV(boo, param_distributions=param_dist,
                                   n_iter=n_iter_search,scoring='log_loss',cv=2,verbose=10)
random_search.fit(X_train, y_train)

report= random_search.grid_scores_
random_search.best_estimator_
param=random_search.best_params_
P=random_search.predict_proba(X_test)
log_loss(y_test, P)

"""
[mean: -0.56698, std: 0.00565, params: {'max_features': 0.23571983882706693, 'n_estimators': 361, 'learning_rate': 0.02, 'max_depth': 5, 'subsample': 0.7}, mean: -0.51726, std: 0.00256, params: {'max_features': 0.26159348375280944, 'n_estimators': 436, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.7}, mean: -0.61361, std: 0.00606, params: {'max_features': 0.6514187354524714, 'n_estimators': 211, 'learning_rate': 0.02, 'max_depth': 5, 'subsample': 0.7}]
"""