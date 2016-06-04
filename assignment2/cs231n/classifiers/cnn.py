import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    
    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    
    (C, H, W) --CRP--> (F, H/2, W/2) --reshape--> (F*H*W/4) -- affine --> relu --> affine --> softmax
    
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                             hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                             dtype=np.float32):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
            of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C, H, W = input_dim
        
        # conv-relu-pool
        self.params['b1'] = np.zeros(num_filters)
        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        
        # hidden affine
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters*H*W / 4, hidden_dim))

        # output affine
        self.params['b3'] = np.zeros(num_classes)
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
         
 
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        
        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # CRP
        activations, crp_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        
        # reshape
        N, F, HH, WW = activations.shape
        activations = activations.reshape((N, F*HH*WW))
        
        # affine2
        activations, affine2_cache = affine_forward(activations, W2, b2)
        
        # relu
        activations, relu_cache = relu_forward(activations)
        
        # affine3
        scores, affine3_cache = affine_forward(activations, W3, b3)
                
        if y is None:
            return scores
        
        loss, grads = 0, {}
        
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))

        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        
        gradients, grads['W3'], grads['b3'] = affine_backward(dscores, affine3_cache)
        gradients = relu_backward(gradients, relu_cache)
        gradients, grads['W2'], grads['b2'] = affine_backward(gradients, affine2_cache)
        gradients = gradients.reshape((N, F, HH, WW))
        _, grads['W1'], grads['b1'] = conv_relu_pool_backward(gradients, crp_cache)
        
        grads['W3'] += self.reg * W3
        grads['W2'] += self.reg * W2
        grads['W1'] += self.reg * W1
        
        return loss, grads
    
    
pass
