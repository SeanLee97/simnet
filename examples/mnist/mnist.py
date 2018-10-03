# -*- coding: utf-8 -*-

import sys
sys.path.append('../..')

import numpy as np
from tqdm import tqdm
from io import BytesIO
import base64
from PIL import Image
from sklearn.metrics import accuracy_score

import simnet.nn as nn
import simnet.optim as optim
import simnet.loss as loss
import simnet.reg as reg
from simnet.pipe import Pipe

def img2base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def dataloader():
    train = np.load('./data/traindata.npy')
    trainlabel = np.load('./data/trainlabellogit.npy')
    val = np.load('./data/valdata.npy')
    vallabel = np.load('./data/vallabellogit.npy')
    return (train,trainlabel,val,vallabel)

class Mnistnet(nn.NN):
    def __init__(self,D_in=28*28, D_hidden=28*28*2, D_out=10):
        super(Mnistnet,self).__init__()

        w1_init = np.random.random((D_in, D_hidden)) * 0.005
        b1_init = np.random.randn(D_hidden) * 0.005
        w2_init = np.random.random((D_hidden, D_out)) * 0.005
        b2_init = np.random.randn(D_out) * 0.005

        self.layers = Pipe(
            nn.Linear(D_in, D_hidden, pretrained=(w1_init,b1_init)),
            nn.ReLU(),
            nn.Linear(D_hidden, D_out, pretrained=(w2_init,b2_init))
        )
        self.criterion = loss.MSE()

    def params(self):
        return self.layers.params()


    def forward(self,*args):
        x = args[0]
        return self.layers.forward(x)


    def backward(self,grad=None):
        grad=self.criterion.backward(grad)
        self.layers.backward(grad)

class Trainer(object):
    def __init__(self, config):
        self.net = Mnistnet()
        self.criterion = self.net.criterion
        if config['optim'] == 'sgd':
            self.optimizer = optim.SGD(self.net.params(), lr=config['lr'], momentum=config['momentum'])
        else:
            self.optimizer = optim.Adam(self.net.params(), lr=config['lr'])
        self.loss = []
        self.val_loss=[]
        self.pred = []
        self.data = dataloader()
        self.train,self.trainlabel,self.val,self.vallabel=self.data
        self.epoches=config['epoches']
        self.valbatches=config['valbatches']

    def train_epoch(self):
        epoches = self.epoches
        train_x, train_y = self.train, self.trainlabel

        val_x,val_y=self.val,self.vallabel
        train_batches = train_x.shape[0]
        val_batches=val_x.shape[0]
        loss_batch=train_batches//500
        for epoch in tqdm(range(epoches)):
            running_loss = 0.0
            for batch in tqdm(range(train_batches)):
                self.optimizer.zero_grad()
                input = train_x[batch:batch + 1]/255
                label = train_y[batch:batch + 1]
                pred = self.net(input)
                loss = self.criterion(pred, label)
                running_loss += loss
                self.net.backward()
                self.optimizer.step()
                if batch % loss_batch==1:
                    self.loss.append(running_loss / batch)
                    print("\nAvg loss", running_loss / batch)

                if batch%self.valbatches==0:
                    tmp_pred=[]
                    pred_label = []
                    for val_idx in range(10):
                        rnd_val_idx = np.random.randint(0,val_batches)
                        valinput = val_x[rnd_val_idx:rnd_val_idx + 1] / 255
                        valpred = self.net(valinput)
                        img=Image.fromarray(val_x[rnd_val_idx:rnd_val_idx + 1].reshape(28,28).astype(np.uint8))
                        img_str=img2base64(img)
                        tmp_pred.append((img_str,valpred.flatten().argmax()))
                        pred_label.append(valpred.flatten().argmax())
                    self.pred=tmp_pred
                    print("\npred", pred_label)
                    print("True", label[0])
                    print("accuracy ", accuracy_score(pred_label, label[0]))

            self.loss=[]

    def get_loss(self):
        idx = np.ndarray.tolist(np.linspace(0, len(self.loss) - 1, len(self.loss)))
        return idx, self.loss,self.val_loss

    def get_pred_mnist(self):
        img_url = {}
        pred={}
        for idx,item in enumerate(self.pred):
            img_url['img' + str(idx)] = item[0]
            pred['label'+str(idx)]=str(item[1])
        return img_url,pred

if __name__=='__main__':
    config={
        'lr':0.0001,
        'optim': 'sgd',
        'momentum':0.9,
        'epoches':3000,
        'valbatches': 100,
    }
    trainmnist=Trainer(config)
    trainmnist.train_epoch()
