# -*- coding: utf-8 -*-

"""正则基类
"""

import numpy as np

class Reg(object):
    pass

class Dropout(Reg):
    def __init__(self, dropout=0.0):
        self.keep_prob = 1.0 - dropout

    def __call__(self, x):
        shape = list(x.shape)
        assert len(shape) == 2
        mask = np.random.rand(shape[0], shape[1]) < self.keep_prob
        x = x * mask
        return x / self.keep_prob
		
