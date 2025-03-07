import torch 

class CaterinaEx1(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2, out_features=2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=2, out_features=2)
        )
        
        self.model_layers = [{"type": "Linear", "parameters": [2,2]},
                             {"type": "ReLU"},
                             {"type": "Linear", "parameters": [2,2]},
                             {"type": "ReLU"},
                             {"type": "Linear", "parameters": [2,2]}]
    
    def forward(self, x):
        x = self.layers(x)
        return x