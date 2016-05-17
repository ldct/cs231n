import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    num_classes = W.shape[1]
    num_train = X.shape[0]
        
    loss = 0.0
    for i in xrange(num_train):
        
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        
        # d(ccs)/dW
        dcss = np.zeros(W.shape)
        dcss[:,y[i]] = X[i]
                
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            
            # d(margin)/dW = d(scores[j])/dW - d(ccs)/dW
            
            if margin > 0:
                
                dmargin = np.zeros(W.shape)
                dmargin[:,j] = X[i]
                dmargin -= dcss
                
                loss += margin
                dW += dmargin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W
    
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    
    num_train = X.shape[0]
    num_classes = W.shape[1]

    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    scores = X.dot(W)
    correct_class_score = scores[range(scores.shape[0]), y]

    predicted_class = np.argmax(scores, axis=1)

    # compute margin, including contribution of 1 from score of correct class
    margin = np.maximum(scores - correct_class_score[np.newaxis].T + 1, 0)
    loss += np.sum(margin)
    
    # subtract contribution of 1
    loss -= num_train
        
    v_scores = scores
    v_ccs = correct_class_score

    for j in xrange(num_classes):

        v_margin = v_scores[:,j] - v_ccs + 1
        
        mask = (v_margin > 0)
        
        m_margin = v_margin[mask]
        m_y = y[mask]
        m_X = X[mask]
        
        
        dW[:, j] += np.sum(m_X, axis=0)
        
        for j in xrange(num_classes):
            
            mask = (m_y == j)
            mm_y = m_y[mask]
            mm_X = m_X[mask]
            
            dW[:, j] -= np.sum(mm_X, axis=0)
                            
            
    loss /= num_train    
    dW /= num_train
    
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg*W
            
        
    return loss, dW
