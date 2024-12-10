import torch 

class FullyConnectedNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )
        # self.model_layers = ["flatten", "linear", "relu", "linear", "relu", "linear"]
        self.model_layers = [{"type": "Flatten"},
                             {"type": "Linear", "parameters": [28*28, 512]},
                             {"type": "ReLU"},
                             {"type": "Linear", "parameters": [512, 512]},
                             {"type": "ReLU"},
                             {"type": "Linear", "parameters": [512, 10]}]
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    