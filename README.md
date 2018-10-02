<p align="center">/ simnet /</p>

为了理解深度学习框架的大致机理，使用numpy实现了一个简单的神经网络框架，主要原理是`链式求导法则`

## 目录结构
- nn：核心的网络包
    - NN: 基础网络类
    - Linear：全连接层
    - Variable：基础参数类
    - ReLU / Sigmoid / Tanh：激活函数
- optim：优化器
    - SGD
    - Adam
- loss：损失函数
    - MSE
    - CrossEntropy
- init：初始化器
    - Normal：高斯分布
    - TruncatedNormal：截断高斯分布
- fn.py：激活函数
- pipe.py：pipeline

## 定义网络
`pytorch`风格，以下以一个简单`BP`网络为例
```python
class MyNet(nn.NN):
    def __init__(self,D_in, D_hidden, D_out):
        super(Mnistnet,self).__init__()

        self.layers = Pipe(
            nn.Linear(D_in, D_hidden),
            nn.ReLU(),
            nn.Linear(D_hidden, D_out)
        )
        self.criterion = loss.MSE()

    def forward(self,*args):
        x = args[0]
        return self.layers.forward(x)

    def backward(self,grad=None):
        grad=self.criterion.backward(grad)
        self.layers.backward(grad)
```

待补充...

## 可视化训练
使用simnet实现了一个简单的BP网络，数据集是 `mnist`，做了可视化训练`demo` , 效果如下图
![bp](https://github.com/SeanLee97/simnet/blob/master/docs/bp.png)

## BUG记录
- Adam的实现：实现倒是不难，只不过经常出现NAN的问题，已修复
- CrossEntropy：交叉熵的导数计算一直困扰了我
- Tanh：使用tanh时有时候会出现NAN的情况，不知道是不是导数计算错误

## refrence
- 此项目的可视化训练参照了[BP-Neural-Network](https://github.com/guyuchao/BP-Neural-Network)的实现，此项目的网络架构也是参照此项目，不过加入了更多的模块，包括`Adam`,`CrossEntropy`等
