# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:06:37 2015

@author: Fernandroid
"""
from sklearn import ensemble 
from sklearn.neighbors import KNeighborsClassifier 
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)
P=knn.predict_proba(X_test)

from sklearn.metrics import log_loss
log_loss(y_test, P)

bagging = ensemble.BaggingClassifier(knn,n_estimators=10, max_samples=0.5, max_features=0.9,verbose=5)
bagging.fit(X_train,y_train)
P=bagging.predict_proba(X_test)

