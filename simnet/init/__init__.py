# -*- coding: utf-8 *-

import numpy as np

class Init(object):
    pass

class Normal(Init):
    def __init__(self, shape, mean=0.0, stddev=1.0):
        super(Normal, self).__init__()

        self.shape = shape
        self.mean = mean
        self.stddev = stddev

    def __call__(self):
        return np.random.normal(loc=self.mean, 
                                scale=self.stddev, 
                                size=self.shape)

class TruncatedNormal(Normal):
    def __call__(self):
        low = self.mean - 2.0*self.stddev
        high = self.mean + 2.0*self.stddev
        val = np.random.normal(loc=self.mean, 
                               scale=self.stddev, 
                               size=self.shape)
        val[val<=low] = 0.0
        val[val>=high] = 0.0
        return val


class Uniform(Init):
    def __init__(self, shape, minval=0.0, maxval=1.0):
        self.shape = shape
        self.minval = minval
        self.maxval = maxval

    def __call__(self):
        return np.random.uniform(self.minval, 
                                 self.maxval, 
                                 self.shape)

class Random(Init):
    def __init__(self, shape):
        self.shape=shape

    def __call__(self):
        return np.random.randn(*self.shape) 

class Zero(Init):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self):
        return np.zeros(self.shape)
