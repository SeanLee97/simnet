# -*- coding: utf-8 -*-

import numpy as np
import simnet.init as ini

class NN(object):
    def __init__(self):
        pass

    def forward(self, *args):
        pass

    def backward(self, grad):
        pass

    def params(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)

class Variable(object):
    def __init__(self, wt, dw, b, db):
        self.wt = wt
        self.dw = dw
        self.b = b
        self.db = db

class Linear(NN):
    def __init__(self, 
                 dim_in, 
                 dim_out, 
                 init=None, 
                 pretrained=None,
                 zero_bias=False):
        super(Linear, self).__init__()

        if isinstance(pretrained, tuple):
            self.wt, self.b = pretrained
        else:
            if not isinstance(init, Init):
               init = ini.Random([dim_in, dim_out])
            self.wt = init()
            if zero_bias:
                self.b = ini.Zero([dim_out])()
            else:
                self.b = ini.Random([dim_out])()

        self.input = None
        self.output = None
        self.dw = ini.Zero(self.wt.shape)()
        self.db = ini.Zero([dim_out])()
        self.variable = Variable(self.wt, self.dw, self.b, self.db)

    def params(self):
        return self.variable

    def forward(self, *args):
        self.input = args[0]
        self.output = np.dot(self.input, self.wt) + self.b
        return self.output

    def backward(self, grad):
        self.db = grad
        self.dw += np.dot(self.input.T, grad)
        grad = np.dot(grad, self.wt.T)

        return grad

class Sigmoid(NN):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.input = None
        self.output = None

    def forward(self, *args):
        self.input = args[0]
        self.output = 1.0 / (1.0 + np.exp(-self.input))
        return self.output

    def backward(self, grad):
        grad *= self.output*(1.0-self.output)
        return grad

class Tanh(NN):
    def __init__(self):
        super(Tanh, self).__init__()
        self.input = None
        self.output = None

    def forward(self, *args):
        self.input = args[0]
        self.output = ((np.exp(self.input) - np.exp(-self.input)) / 
                        np.exp(self.input) + np.exp(-self.input))
        return self.output

    def backward(self, grad):
        grad *= 1.0 - np.power(self.output, 2)
        return grad

class ReLU(NN):
    def __init__(self):
        super(ReLU, self).__init__()
        self.input = None
        self.output = None

    def forward(self, *args):
        x = args[0]
        self.input = x
        x[self.input<=0] *= 0
        self.output = x
        return self.output

    def backward(self, grad):
        grad[self.input>0] *= 1.0
        grad[self.input<=0] *= 0.0
        return grad