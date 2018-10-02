# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from simnet.init import *

print("normal")
norm = Normal([4, 3])
print(norm())

print("truncated normal")
norm = TruncatedNormal([4, 3], mean=-0.5)
print(norm())

print("uniform")
uni = Uniform([4, 3])
print(uni())

print("random")
random = Random([4, 3])
print(random())

print("zero")
zero = Zero([3])
print(zero())
