# -*- coding: utf-8 -*-

import numpy as np

def softmax(logit):
    return np.exp(logit) / np.sum(np.exp(logit, 1).reshape(-1, 1))

def log_softmax(logit):
    return np.log(softmax(logit))

def sigmoid(logit):
    return 1.0 / (1.0 + np.exp(-logit))

def relu(logit):
    return np.maximum(logit, 0)

def leaky_relu(logit, alpha=0.3):
    return np.maximum(logit, alpha*x)

