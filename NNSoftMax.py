# -*- coding: utf-8 -*-
"""
@author: Fernandroid
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 04 12:08:27 2015

@author: u304479
"""

def NNSoftMax(nn_params,input_layer_size,hidden_layer_size,num_labels,X, label,landa):                                    
    """
    NNSoftMax Implements the neural network cost function for a 2 layer
    neural network which performs classification
    [J grad] = DropoutNN2hl(nn_params, hidden_layer_size, num_labels, ...
     X, y, lambda) computes the cost and gradient of the neural network. The
     parameters for the neural network are "unrolled" into the vector
     nn_params and need to be converted back into the weight matrices. 
     The returned parameter grad should be a "unrolled" vector of the
     partial derivatives of the neural network.
     Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
     for our 3 layer neural network
     P: is a row vector (array(1,2)) for Bernoulli variable with P(1) equals p for input layer 
     and P(2) equals for hidden layer
     X: data array(n samples,fetures)
     num_labels: number of classes
     label: sample classes array(size(m))
     """
    import numpy as np
    from scipy import sparse  
     
  
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size+1 )], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],(num_labels, hidden_layer_size + 1))
    """input_layer_size=300
    hidden_layer_size=500
    num_labels=10
    X=random.rand(1000,300)
    Theta1=random.rand(hidden_layer_size,input_layer_size+1)*2*0.01-0.01
    Theta2=random.rand(num_labels,hidden_layer_size+1)*2*0.01-0.01 
    label=random.random_integers(num_labels,size=(m))-1
    """
 
    
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


    return (J, grad)

