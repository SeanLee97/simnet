# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

import numpy as np
from simnet.reg import Dropout

x = np.random.normal(size=[4,2])
rate = 0.5
dropout = Dropout(dropout=rate)
print("raw matrix")
print(x)
print("after dropout with rate %f" % rate)
print(dropout(x))

