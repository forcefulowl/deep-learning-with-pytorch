import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

im = Image.open('sample.jpeg').convert('L')
im = np.array(im, dtype = 'float32')


im = torch.Tensor(im.reshape(1,1,im.shape[0],im.shape[1]))

conv1 = nn.Conv2d(1,1,3,bias = False)

kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
kernel = kernel.reshape((1,1,3,3))
conv1.weight.data = torch.Tensor(kernel)

result = conv1(Variable(im))
result = result.data.squeeze().numpy()

# plt.imshow(result, cmap = 'gray')
# plt.show()

pool1 = nn.MaxPool2d(2,2)
small = pool1(Variable(im))
small = small.data.squeeze().numpy()

plt.imshow(small,cmap='gray')
plt.show()
