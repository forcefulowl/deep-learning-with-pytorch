import torch
import random
from torch.autograd import Variable

class Net(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(Net,self).__init__()
        self.linear1 = torch.nn.Linear(input_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, output_channels)

    def forward(self,x):
        """
        In the forward function we accept a Variable of input data and we must return a Variable of output data.
        We can use Modules defined in the constructor as well as arbitrary operators on Variables.
        """
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred


batch_size, input_channels,hidden_channels,output_channels = 64,1000,100,10

x = Variable(torch.randn(batch_size,input_channels))
y = Variable(torch.randn(batch_size,output_channels))

model = Net(input_channels,hidden_channels,output_channels)


criterion = torch.nn.MSELoss()

# Use the optim package to define an Optimizer that will update the weights of the model.
# The first argument to the Adam constructor tells the optimizer which Variables it should update
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for t in range(500):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)

    # compute and print loss. We pass Variables containing the predicted and true values of y,
    # and the loss function returns a Variable containing the loss
    loss = criterion(y_pred, y)
    print(t, loss.data[0])

    # Before backward pass, user the optimizer object to zero all of the gradients for the vairables it will update
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable parameters of the model.
    loss.backward()

    # Updating parameters
    optimizer.step()



