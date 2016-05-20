import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    for i in xrange(num_train):
        
        scores = X[i].dot(W)
        scores -= np.max(scores)
        
        unnormalized_prob = np.exp(scores)
        normalization_sum = np.sum(unnormalized_prob)

        loss -= np.log(unnormalized_prob[y[i]] / normalization_sum)
        
        dns = np.zeros_like(W)
        
        for j in range(num_classes):
            dns[:,j] += np.exp(scores[j]) * X[i]
            
        dW += dns / normalization_sum
        dW[:,y[i]] -= X[i]
                    
    loss /= num_train
    dW /= num_train
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW1 = np.zeros_like(W)
    dW2 = np.zeros_like(W)

    scores = X.dot(W)
    scores -= np.max(scores)
    
    unnormalized_prob = np.exp(scores)
    normalization_sum = np.sum(unnormalized_prob, axis=1)
    
    correct_class_unp = unnormalized_prob[range(unnormalized_prob.shape[0]), y]
    
    losses = -np.log(correct_class_unp / normalization_sum)
        
    loss = sum(losses) / num_train

    
    for j in xrange(num_classes):
        dW1[:,j] -= np.sum(X[y == j], axis=0)
    
    for j in range(num_classes):
        weight = np.exp(scores[:,j]) / normalization_sum

        weightedX = np.multiply(weight[:,np.newaxis], X)

        dW2[:,j] += np.sum(weightedX, axis=0)
                            
    dW = (dW1 + dW2) / num_train
        
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W
            
    return loss, dW

