import torch 
import torchvision
import torch.utils
from torch.utils.data import DataLoader
from torchvision import datasets


from FullyConnectedNet import FullyConnectedNet
from LeNet import LeNet
from AlexNet import AlexNet
from VGGNet import VGGNet


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
learning_rate = 1e-3
batch_size = 64
epochs = 30
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
    filename = f"{model_type}_{dataset}.pth"
    torch.save(state, filename)

    return 


if __name__ == "__main__":
    # dataset_list = ["MNIST", "CIFAR10"]
    dataset_list = ["CIFAR10"]
    # model_list = ["FullyConnectedNet", "LeNet", "AlexNet", "VGGNet"]
    model_list = ["AlexNet", "VGGNet"]
    for d in dataset_list:
        for m in model_list:
            dataset = d
            model_type = m
            main()