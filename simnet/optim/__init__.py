# -*- coding: utf-8 -*-

import numpy as np 

class Optim(object):
    def __init__(self, params, lr=1e-2, *args, **kwargs):
        self.params = params
        self.lr = lr

    def zero_grad(self):
        for params in self.params:
            params.dw *= 0
            params.db *= 0

    def step(self):
        pass

    def __call__(self):
        self.step()

class SGD(Optim):
    def __init__(self, params, lr=1e-2, momentum=0.9):
        super(SGD, self).__init__(params, lr)
        self.momentum = momentum

    def step(self):
        for params in self.params:
            #params.wt -= params.wt*self.momentum - self.lr*params.dw
            params.wt -= self.lr*params.dw
            params.b -= self.lr * params.db   

class Adam(Optim):
    def __init__(self, params, lr=1e-2, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):
        super(Adam, self).__init__(params, lr)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.wt_ms = [np.zeros(params.wt.shape) for params in self.params]
        self.b_ms  = [np.zeros(params.b.shape)  for params in self.params]
        self.wt_vs = [np.zeros(params.wt.shape) for params in self.params]
        self.b_vs  = [np.zeros(params.b.shape)  for params in self.params]

    def step(self):
        for i, params in enumerate(self.params):
            wt_m = self.wt_ms[i]
            b_m = self.b_ms[i]
            wt_v = self.wt_vs[i]
            b_v = self.b_vs[i]

            wt_m_t = (self.beta_1 * wt_m) + (1.0 - self.beta_1) * params.dw
            b_m_t = (self.beta_1 * b_m) + (1.0 - self.beta_1) * params.db
            wt_v_t = (self.beta_2 * wt_v) + (1.0 - self.beta_2) * params.dw
            b_v_t = (self.beta_2 * b_v) + (1.0 - self.beta_2) * params.db

            wt_m_t[wt_m_t<0] = 0
            b_m_t[b_m_t<0] = 0
            wt_v_t[wt_v_t<0] = 0
            b_v_t[b_v_t<0] = 0

            params.wt -= self.lr*wt_m_t / (np.sqrt(wt_v_t) + self.epsilon)
            params.b -= self.lr*b_m_t / (np.sqrt(b_v_t) + self.epsilon)

            self.wt_ms[i] = wt_m_t
            self.b_ms[i] = b_m_t
            self.wt_vs[i] = wt_v_t
            self.b_vs[i] = b_v_t