import torch


class ProjectionNet(torch.nn.Module):
    def __init__(self):
        super(ProjectionNet, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(256, 128),
            # torch.nn.ReLU(),
            # torch.nn.Linear(128, 128)
        )

    def forward(self, x):
        output = self.fc(x)
        return output


class TripletNet(torch.nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x, p, n):
        output1 = self.embedding_net(x)
        output2 = self.embedding_net(p)
        output3 = self.embedding_net(n)
        return output1, output2, output3


def xavier_initialize(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


# class ConvolutionalAutoEncoder(torch.nn.Module):
#     def __init__(self):
#         super(ConvolutionalAutoEncoder, self).__init__()
#         ## encoder layers ##
#         # conv layer (depth from 3 --> 16), 3x3 kernels
#         self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
#         # conv layer (depth from 16 --> 4), 3x3 kernels
#         self.conv2 = torch.nn.Conv2d(16, 4, 3, padding=1)
#         # pooling layer to reduce x-y dims by two; kernel and stride of 2
#         self.pool = torch.nn.MaxPool2d(2, 2)
#
#         ## decoder layers ##
#         ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#         self.t_conv1 = torch.nn.ConvTranspose2d(4, 16, 2, stride=2)
#         self.t_conv2 = torch.nn.ConvTranspose2d(16, 3, 2, stride=2)
#
#     def forward(self, x):
#         ## encode ##
#         # add hidden layers with relu activation function
#         # and maxpooling after
#         x = torch.nn.functional.relu(self.conv1(x))
#         x = self.pool(x)
#         # add second hidden layer
#         x = torch.nn.functional.relu(self.conv2(x))
#         x = self.pool(x)  # compressed representation
#
#         ## decode ##
#         # add transpose conv layers, with relu activation function
#         x = torch.nn.functional.relu(self.t_conv1(x))
#         # output layer (with sigmoid for scaling from 0 to 1)
#         x = torch.nn.functional.sigmoid(self.t_conv2(x))
#
#         return x

class ConvolutionalAutoEncoder(torch.nn.Module):
    def __init__(self):
        super(ConvolutionalAutoEncoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_conv4 = torch.nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.t_conv3 = torch.nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.t_conv2 = torch.nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.t_conv1 = torch.nn.ConvTranspose2d(16, 3, 2, stride=2)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool(x)  # compressed representation
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        # x = torch.nn.functional.relu(self.conv4(x))
        # x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        # x = torch.nn.functional.relu(self.t_conv4(x))
        x = torch.nn.functional.relu(self.t_conv3(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.nn.functional.relu(self.t_conv2(x))
        x = torch.nn.functional.sigmoid(self.t_conv1(x))

        return x