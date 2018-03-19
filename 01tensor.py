import torch
import numpy as np
from torch.autograd import Variable
import torch

numpy_tensor = np.random.randn(2,3)
# print(numpy_tensor)
pytorch_tensor = torch.Tensor(numpy_tensor)
# print(pytorch_tensor.shape)
# print(pytorch_tensor.type())
# print(pytorch_tensor.dim())
# print(pytorch_tensor.numel())

x = torch.ones(2,3)
x = x.long()
x = x.float()
x = torch.randn(4,3)

max_value, max_idx= torch.max(x,dim=1)
sum_x = torch.sum(x, dim=1)

x = torch.randn(2,3)
y = torch.randn(2,3)
x = Variable(x, requires_grad = True)
y = Variable(y, requires_grad = True)

z = torch.sum(x+y)

z.backward()

x = Variable(torch.FloatTensor([2]),requires_grad = True)
y = x**2
y.backward()

x = Variable(torch.Tensor([2]), requires_grad = True)
y = x+2
z = y**2 + 3
z.backward()

x = Variable(torch.randn(10, 20), requires_grad=True)
y = Variable(torch.randn(10, 5), requires_grad=True)
w = Variable(torch.randn(20, 5), requires_grad=True)
out = torch.mean(y-torch.matmul(x,w))
out.backward()

x = Variable(torch.FloatTensor([3]),requires_grad = True)
y = x*2 + x**2 + 3
y.backward(retain_graph=True)

y.backward()


x = torch.randn(2,3)
print(x)
x = np.array(x, dtype = 'float32')/255
print('==')
print(x)
x = x.reshape((-1,))
print('==')
print(x)