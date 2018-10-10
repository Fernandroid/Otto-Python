# -*- coding: utf-8 -*-
"""
@author: Fernandroid
"""
from scipy.optimize import minimize
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
"""This function is the one that will be minimized with respect to w,
    the model weights"""
def fun(w,probs,y_true):
    sum = 0
    for i in range(len(probs)):
        sum+= probs[i]*w[i]
    return log_loss(y_true,sum)
    
## First save your model probabilities:
from sklearn.externals import joblib
X_train=joblib.load('...\Otto_group\X_train.pkl')
y_train=joblib.load('...\Otto_group\y_train.pkl')
X_test=joblib.load('...\Otto_group\X_test.pkl')
y_test=joblib.load('...\Otto_group\y_test.pkl')
#Model
boo=joblib.load('...\Otto_group\GBCtotale3.pkl')
boo2=joblib.load('...\Otto_group\GBCtotale.pkl')
NN = pd.read_csv('...\Otto_group\ModelNNrelu.csv')
#Predictions
P=boo.predict_proba(X_test)
P2=boo2.predict_proba(X_train)
P2=NN.values
probs=[P,P2]
## w0 is the initial guess for the minimum of function 'fun'
## This initial guess is that all weights are equal
w0 =np.ones(len(probs))/(len(probs))

## This sets the bounds on the weights, between 0 and 1
bnds = tuple((0,1) for w in w0)
## This sets the constraints on the weights, they must sum to 1
## Or, in other words, 1 - sum(w) = 0
cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
""" ---------------------------------------------------------------------------
fun is the function defined above
w0 is the initial estimate of weights
(probs, y_true) are the additional arguments passed to 'fun'; probs are the probabilities,
y_true is the expected output for your training set
method = 'SLSQP' is a least squares method for minimizing the function;
----------------------------------------------------------------------------"""
option = {"maxiter": 100,"disp":1}
F= minimize(fun,w0,(probs,y_train),method='L-BFGS-B',jac=0,bounds=bnds,constraints=cons,options=option) 
weights=F['x']
## As a sanity check, make sure the weights do in fact sum to 1
print("Weights sum to %0.4f:" % weights.sum())
## Print out the weights
print(weights)
## This will combine the model probabilities using the optimized weights
y_prob = 0
for i in range(len(probs)):
    y_prob += probs[i]*weights[i]

log_loss(y_train,P)
log_loss(y_train,P2)
log_loss(y_train,y_prob)

