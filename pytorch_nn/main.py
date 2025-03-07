import torch 
import torchvision
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from FullyConnectedNet import FullyConnectedNet
from fc_5x100 import fc_5x100
from LeNet import LeNet
from AlexNet import AlexNet
from VGGNet import VGGNet
from toy_example import ToyNet
from caterina_ex1 import CaterinaEx1
from caterina_ex2 import CaterinaEx2
from deeppoly_ex import DeepPolyEx


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
learning_rate = 1e-3
batch_size = 64
epochs = 100 
dataset = "MNIST"
num_classes = 10
model_type = "LeNet"
filename = ""
state = {}
vgg_arch16 = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size 
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main():
    # load data
    training_data: torchvision.datasets
    testing_data: torchvision.datasets
    if dataset == "MNIST":
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        testing_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
    elif dataset == "CIFAR10":
        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )
        testing_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)

    # fully connected network
    if model_type == "FullyConnectedNet":
        model = FullyConnectedNet().to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(model)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")

        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }
    elif model_type == "fc_5x100":
        model = fc_5x100().to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(model)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")

        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }
    elif model_type == "LeNet":
        model = LeNet().to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(model)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")

        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }   
    elif model_type == "AlexNet":
        model = AlexNet(num_classes=num_classes).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(model)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")

        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }
    elif model_type == "VGGNet":
        model = VGGNet(vgg_arch=vgg_arch16, num_classes=num_classes).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        print(model)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")

        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }
    elif model_type == "ToyNet":
        model = ToyNet()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        with torch.no_grad():
            fc1_weight = torch.tensor([[0.8, -0.7],[0.6, 0.5]], dtype=torch.float)
            model.layers[0].weight = torch.nn.Parameter(fc1_weight)
            model.layers[0].bias = torch.nn.Parameter(torch.zeros(2))
            
            fc2_weight = torch.tensor([[-1, 0.4]], dtype=torch.float)
            model.layers[2].weight = torch.nn.Parameter(fc2_weight)
            model.layers[2].bias = torch.nn.Parameter(torch.zeros(1)) 
            
        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }
    elif model_type == "ToyNetNeg":
        model = ToyNet()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        with torch.no_grad():
            fc1_weight = torch.tensor([[-0.8, -0.7],[-0.6, -0.5]], dtype=torch.float)
            model.layers[0].weight = torch.nn.Parameter(fc1_weight)
            model.layers[0].bias = torch.nn.Parameter(torch.zeros(2))
            
            fc2_weight = torch.tensor([[-1, -0.4]], dtype=torch.float)
            model.layers[2].weight = torch.nn.Parameter(fc2_weight)
            model.layers[2].bias = torch.nn.Parameter(torch.zeros(1))
        
        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }
    elif model_type == "CaterinaEx1":
        model = CaterinaEx1()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        with torch.no_grad():
            fc1_weight = torch.tensor([[1, 1],[1, -1]], dtype=torch.float)
            model.layers[0].weight = torch.nn.Parameter(fc1_weight)
            model.layers[0].bias = torch.nn.Parameter(torch.zeros(2))
            
            fc2_weight = torch.tensor([[1, 1], [1, -1]], dtype=torch.float)
            model.layers[2].weight = torch.nn.Parameter(fc2_weight)
            model.layers[2].bias = torch.nn.Parameter(torch.zeros(2))
            
            fc3_weight = torch.tensor([[1, 1], [0, 1]], dtype=torch.float)
            model.layers[4].weight = torch.nn.Parameter(fc3_weight)
            model.layers[4].bias = torch.nn.Parameter(torch.tensor([1, -1.25]))
        
        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }
    elif model_type == "CaterinaEx2":
        model = CaterinaEx2()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        with torch.no_grad():
            fc1_weight = torch.tensor([[1, 1],[0.5, 0.5]], dtype=torch.float)
            fc1_bias = torch.tensor([4,3], dtype=torch.float)
            model.layers[0].weight = torch.nn.Parameter(fc1_weight)
            model.layers[0].bias = torch.nn.Parameter(fc1_bias)
            
            fc2_weight = torch.tensor([[2, 3], [1, -1]], dtype=torch.float)
            model.layers[2].weight = torch.nn.Parameter(fc2_weight)
            model.layers[2].bias = torch.nn.Parameter(torch.zeros(2))
            
            fc3_weight = torch.tensor([[1, -1], [0.5, -1.5]], dtype=torch.float)
            fc3_bias = torch.tensor([-14, -8], dtype=torch.float)
            model.layers[4].weight = torch.nn.Parameter(fc3_weight)
            model.layers[4].bias = torch.nn.Parameter(fc3_bias)
            
            fc4_weight = torch.tensor([[0.5, -2], [0, 1]], dtype=torch.float)
            fc4_bias = torch.tensor([1, 0], dtype=torch.float)
            model.layers[6].weight = torch.nn.Parameter(fc4_weight)
            model.layers[6].bias = torch.nn.Parameter(fc4_bias)
        
        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }
    elif model_type == "DeepPolyEx":
        model = DeepPolyEx()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        with torch.no_grad():
            fc1_weight = torch.tensor([[1, 1],[1, -1]], dtype=torch.float)
            model.layers[0].weight = torch.nn.Parameter(fc1_weight)
            model.layers[0].bias = torch.nn.Parameter(torch.zeros(2))
            
            fc2_weight = torch.tensor([[1, 1], [1, -1]], dtype=torch.float)
            model.layers[2].weight = torch.nn.Parameter(fc2_weight)
            model.layers[2].bias = torch.nn.Parameter(torch.zeros(2))
            
            fc3_weight = torch.tensor([[1, 1], [0, 1]], dtype=torch.float)
            model.layers[4].weight = torch.nn.Parameter(fc3_weight)
            model.layers[4].bias = torch.nn.Parameter(torch.tensor([1, 0], dtype=torch.float))
        
        # save the model into pth file.
        state = {
            'epoch': epochs,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_layers': model.model_layers,
        }
         
    # filename = f"./trained_models/{model_type}_{dataset}.pth"
    filename = f"./trained_models/{model_type}.pth"
    torch.save(state, filename)

    return 


def record_input_id(dataset_name: str, label: int):
    if dataset_name == "mnist":
        dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))  
    elif dataset_name == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transforms.Compose([transforms.ToTensor()]))
    
    # record the input id by specific label
    with open(f"{dataset_name}_{label}.txt", "w+") as f:
        for i, d in enumerate(dataset):
            # d := tuple(tensor, int) := (image, label)
            if d[1] == label: 
                f.write(f"{i}\n")

    print("finished!")
    return 



if __name__ == "__main__":
    # dataset_list = ["MNIST", "CIFAR10"]
    dataset_list = ["MNIST"]
    model_list = ["fc_5x100"]
    # model_list = ["FullyConnectedNet", "LeNet", "AlexNet", "VGGNet"]
    # model_list = ["AlexNet", "VGGNet"]
    # model_list = ["ToyNet", "ToyNetNeg", "CaterinaEx1", "CaterinaEx2", "DeepPolyEx"]
    for d in dataset_list:
        for m in model_list:
            dataset = d
            model_type = m
            main()
    
    
    # record_input_id("mnist", 0)