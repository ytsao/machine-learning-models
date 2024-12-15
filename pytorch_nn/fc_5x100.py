import torch 
import numpy as np

class Normalization(torch.nn.Module):

    def __init__(self, device, mean=0.1307, sigma=0.3081):
        super(Normalization, self).__init__()
        # self.mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
        # self.sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)
        mean = np.array(mean) if isinstance(mean, list) else np.array([mean])
        sigma = np.array(sigma) if isinstance(sigma, list) else np.array([sigma])

        self.mean = torch.nn.Parameter(torch.FloatTensor(
            mean).view((1, -1, 1, 1)), False)
        self.sigma = torch.nn.Parameter(torch.FloatTensor(
            sigma).view((1, -1, 1, 1)), False)

    def forward(self, x):
        return (x - self.mean) / self.sigma

class fc_5x100(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalization = Normalization(device="cpu", mean=0.1307, sigma=0.3081)
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(28*28, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 10)
        )
        
        self.model_layers = [{"type": "Normalization", "weights": [0.1307, 0.3081]},
                             {"type": "Flatten"},
                             {"type": "Linear", "parameters": [28*28,100]},
                             {"type": "ReLU"},
                             {"type": "Linear", "parameters": [100,100]},
                             {"type": "ReLU"},
                             {"type": "Linear", "parameters": [100,100]},
                             {"type": "ReLU"},
                             {"type": "Linear", "parameters": [100,100]},
                             {"type": "ReLU"},
                             {"type": "Linear", "parameters": [100,100]},
                             {"type": "ReLU"},
                             {"type": "Linear", "parameters": [100,10]}]
    
    def forward(self, x):
        x = self.normalization(x)
        logits = self.layers(x)
        return logits