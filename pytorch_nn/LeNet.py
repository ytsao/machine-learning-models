import torch 

"""
Varient of fully-connected neural network:
different number of layers, different number of neurons in each layer, different activation functions, etc.
1. number of layers: 5, 10, 20, 30
2. 
3. 
4.
5. 
"""
class FullyConnectedNN5(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNN5, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FullyConnectedNN10(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNN10, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FullyConnectedNN20(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNN20, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc4 = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class FullyConnectedNN30(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnectedNN30, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc3 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc4 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.fc5 = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x


"""
Well-known CNN architectures:
1. LeNet
2. AlexNet
3. VGGNet
4. ResNet
5. DenseNet
6. GoogleNet
"""
class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = torch.nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 16*5*5) # flatten
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AlexNet(torch.nn.Module):
    def __init__(self) -> None:
        super(AlexNet, self).__init__()
    
    def forward(x):
        return x
    
    
class VGGNet(torch.nn.Module):
    def __init__(self) -> None:
        super(VGGNet, self).__init__()
    
    def forward(x):
        return x


class ResNet(torch.nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()
    
    def forward(x):
        return x


class DenseNet(torch.nn.Module):
    def __init__(self) -> None:
        super(DenseNet, self).__init__()
    
    def forward(x):
        return x
    

class GoogleNet(torch.nn.Module):
    def __init__(self) -> None:
        super(GoogleNet, self).__init__()
    
    def forward(x):
        return x


