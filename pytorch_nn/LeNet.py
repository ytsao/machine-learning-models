import torch 

class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = torch.nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = torch.nn.Linear(in_features=120, out_features=84)
        self.fc3 = torch.nn.Linear(in_features=84, out_features=10)

        self.model_layers = ["conv", "relu", "max_pool", "conv", "relu", "max_pool", 
                             "flatten", "linear", "relu", "linear", "relu", "linear"]
    
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
