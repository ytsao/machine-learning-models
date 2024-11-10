import torch 

class AlexNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, padding=2, stride=4)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2)
        self.conv3 = torch.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.fc1 = torch.nn.Linear(in_features=256*6*6, out_features=4096)
        self.fc2 = torch.nn.Linear(in_features=4096, out_features=1024)
        self.fc3 = torch.nn.Linear(in_features=1024, out_features=num_classes)

        self.model_layers = []

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2)
        # x = x.view(-1, 256*6*6)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, p=0.5)
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.dropout(x, p=0.5)
        x = self.fc3(x)
        return x