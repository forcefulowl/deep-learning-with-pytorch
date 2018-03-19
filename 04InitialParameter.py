import numpy as np
import torch
from torch import nn
from torch.nn import init

net1 = nn.Sequential(
    nn.Linear(30, 40),
    nn.ReLU(),
    nn.Linear(40, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)
w1 = net1[0].weight
b1 = net1[0].bias
net1[0].weight.data = torch.from_numpy(np.random.uniform(3, 5, size=(40, 30)))

class sim_net(nn.Module):
    def __init__(self):
        super(sim_net, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(30, 40),
            nn.ReLU()
        )

        self.l1[0].weight.data = torch.randn(40, 30)

        self.l2 = nn.Sequential(
            nn.Linear(40, 50),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Linear(50, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


net2 = sim_net()

for i in net2.children():
    print(i)

print('====')

for i in net2.modules():
    print(i)

for layer in net2.modules():
    if isinstance(layer, nn.Linear):
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size = param_shape))