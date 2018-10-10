# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:44:01 2015

@author: u304479
"""

""" Neronal Network """
import numpy as np
from scipy import sparse  
from scipy.optimize import fmin_bfgs
from scipy.optimize import minimize

X=X_train[:100,:]
Y=y_train[:100]
input_layer_size=93
hidden_layer_size=100
num_labels=9
landa=0
Theta1=np.random.rand(hidden_layer_size,input_layer_size+1)*2*(1./np.sqrt(input_layer_size+1))-(1./np.sqrt(input_layer_size+1))
Theta2=np.random.rand(num_labels,hidden_layer_size+1)*2*(1./np.sqrt(hidden_layer_size+1))-(1./np.sqrt(hidden_layer_size+1))
nn_params=np.concatenate((Theta1.flatten(),Theta2.flatten()))

def NNSoftMaxCost(nn_params,input_layer_size,hidden_layer_size,num_labels,X, label,landa):                                    


    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size+1 )], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],(num_labels, hidden_layer_size + 1))
 
    def sigmoid(z):
        s=1/(1+np.exp(-z))
        return s
    
    def sigmoidGradient(z):
        s=sigmoid(z)*(1-sigmoid(z))
        return s
    
    def SoftMax(z):
        maxes = np.amax(z, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(z - maxes)
        P = e /np.sum(e, axis=0)
        return P
    
    "Setup some useful variables"

    m = X.shape[0]
    J = 0.
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))
    Y=sparse.csc_matrix((np.ones((m)),(np.arange(m),label)),shape=(m,num_labels)).toarray()

    " Part 1: forward propagation:"

    X = np.hstack([np.ones((m, 1)), X])
    z2=np.dot(X,Theta1.T)
    a2=sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2])
    z3=np.dot(a2,Theta2.T)
    h=SoftMax(z3.T)

    " Cost function"
    J=-(1./m)*np.sum(Y*np.log(h.T)) + (landa/(2.*m))*(np.sum(Theta1[:,1:]**2)+np.sum(Theta2[:,1:]**2))
    return J



def NNSoftMaxGrad(nn_params,input_layer_size,hidden_layer_size,num_labels,X, label,landa):
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size+1 )], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],(num_labels, hidden_layer_size + 1))
 
    
    def sigmoid(z):
        s=1/(1+np.exp(-z))
        return s
    
    def sigmoidGradient(z):
        s=sigmoid(z)*(1-sigmoid(z))
        return s
    
    def SoftMax(z):
        maxes = np.amax(z, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        e = np.exp(z - maxes)
        P = e /np.sum(e, axis=0)
        return P
    
    "Setup some useful variables"

    m = X.shape[0]
    J = 0.
    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))
    Y=sparse.csc_matrix((np.ones((m)),(np.arange(m),label)),shape=(m,num_labels)).toarray()

    " Part 1: forward propagation:"

    X = np.hstack([np.ones((m, 1)), X])
    z2=np.dot(X,Theta1.T)
    a2=sigmoid(z2)
    a2 = np.hstack([np.ones((m, 1)), a2])
    z3=np.dot(a2,Theta2.T)
    h=SoftMax(z3.T)

    " Cost function"
    #J=-(1./m)*np.sum(Y*np.log(h.T)) + (landa/(2.*m))*(np.sum(Theta1[:,1:]**2)+np.sum(Theta2[:,1:]**2))
  

    "Part 2: back propagation to compute gradient"
   
    delta3=(h.T-Y)
    delta2=np.dot(delta3,Theta2[:,1:])*sigmoidGradient(z2)

    Theta2_grad=np.dot(delta3.T,a2)/m
    Theta1_grad=np.dot(delta2.T,X)/m
    b2=Theta2_grad[:,0]
    b1=Theta1_grad[:,0]
    Theta2_grad=np.hstack([b2.reshape(b2.shape[0],1), Theta2_grad[:,1:] + (landa/m)*Theta2[:,1:]])
    Theta1_grad=np.hstack([b1.reshape(b1.shape[0],1), Theta1_grad[:,1:] + (landa/m)*Theta1[:,1:]])

    " Unroll gradients"
    grad=np.concatenate((Theta1_grad.flatten(),Theta2_grad.flatten()))
    return grad



J= NNSoftMaxCost(nn_params,input_layer_size,hidden_layer_size,num_labels,X, Y,landa)
G= NNSoftMaxGrad(nn_params,input_layer_size,hidden_layer_size,num_labels,X, Y,landa)
def decorated_cost(nn_params):
    return NNSoftMaxCost(nn_params,input_layer_size,hidden_layer_size,num_labels,X, Y,landa)
    
def decorated_grad(nn_params):
    return NNSoftMaxGrad(nn_params,input_layer_size,hidden_layer_size,num_labels,X, Y,landa)

sol=fmin_bfgs(decorated_cost,nn_params,fprime=decorated_grad,  maxiter=100,disp=1)


def decorated_cost(nn_params):
    return NNSoftMax(nn_params,input_layer_size,hidden_layer_size,num_labels,X, Y,landa)

option = {"maxiter": 100,"disp":1}
sol=minimize(decorated_cost,nn_params,method='L-BFGS-B',jac=0,options=option )
dpa=sol.x


scipy.optimize.check_grad(func, grad, x0, *args, **kwargs)