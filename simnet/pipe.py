# -*- coding: utf-8 -*-

from .nn import Linear

class Pipe(object):
    def __init__(self, *layers):
        super(Pipe, self).__init__()
        self.layers = []
        self.parameters = []
        for layer in layers:
            self.layers.append(layer)
            if isinstance(layer, Linear):
                self.parameters.append(layer.params())

    def add_layer(self, layer):
        self.layers.append(layer)
        if isinstance(layer, Linear):
            self.parameters.append(layer.params())

    def forward(self, *args):
        x = args[0]
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for idx in range(len(self.layers)-1, -1, -1):
            grad = self.layers[idx].backward(grad)
        
    def params(self):
        return self.parameters
