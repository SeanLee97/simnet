# -*- coding: utf-8 -*-

import numpy as np
import simnet.fn as fn

class Loss(object):
    def __init__(self):
        pass

    def forward(self, *args):
        pass

    def backward(self, grad=None):
        pass

    def __call__(self, *args):
        return self.forward(*args)

class MSE(Loss):
    def __init__(self):
        super(MSE, self).__init__()
        self.label = None
        self.logit = None
        self.grad = None
        self.loss = None

    def forward(self, logit, label):
        self.logit, self.label = logit, label
        self.loss = np.sum(0.5*np.square(self.logit - self.label))
        return self.loss

    def backward(self, grad=None):
        self.grad = self.logit - self.label
        grad_ = np.sum(self.grad, axis=0)
        return np.expand_dims(grad_, axis=0)

class CrossEntropy(Loss):
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.label = None
        self.logit = None
        self.grad = None
        self.loss = None

    def forward(self, logit, label):
        self.logit, self.label = logit, label
        self.loss = -np.sum(np.multiply(label, fn.log_softmax(logit)), 1)
        return self.loss

    def backward(self, logit, label):
        self.grad = fn.softmax(logit) - label
        grad_ = np.sum(self.grad, axis=0)
        return np.expand_dims(grad_, axis=0)
