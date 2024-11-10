import torch 

class VGGNet(torch.nn.Module):
    def __init__(self, vgg_arch, num_classes):
        super(VGGNet, self).__init__()
        self.features = self.vgg_block(vgg_arch)

        self.fc1 = torch.nn.Linear(in_features=512*7*7, out_features=4096)
        self.fc2 = torch.nn.Linear(in_features=4096, out_features=1024)
        self.fc3 = torch.nn.Linear(in_features=1024, out_features=num_classes)

    def vgg_block(self, vgg_arch):
        layers = []
        in_channels = 3

        for v in vgg_arch:
            if v == "M":
                layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1))
                layers.append(torch.nn.ReLU())
                in_channels = v
        return torch.nn.Sequential(*layers)   

    def forward(self, x):
        x = self.features(x)

        # x = x.view(x.size(0), -1)
        x = torch.flatten(x, 1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x