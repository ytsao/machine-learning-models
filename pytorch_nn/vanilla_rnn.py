import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader


mnist_train = datasets.MNIST(
    "./mnist_train", train=True, download=True, transform=T.Compose([T.ToTensor()])
)
mnist_test = datasets.MNIST(
    "./mnist_test", train=False, download=True, transform=T.Compose([T.ToTensor()])
)
mnist_train, mnist_val = torch.utils.data.random_split(mnist_train, [50000, 10000])
mini_trains = DataLoader(mnist_train, batch_size=128, shuffle=True)
mini_vals = DataLoader(mnist_val, batch_size=128, shuffle=True)
mini_tests = DataLoader(mnist_test, batch_size=128, shuffle=True)


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x => [n, 28, 28] => [batch_size, sequence_length, input_size]
        output, _ = self.lstm(x)
        # output => [n, 28, 64] => [batch_size, seqence_length, hidden_size]
        output = output[:, -1, :]

        return self.fc(output)


model = RNN(input_size=28, hidden_size=64, output_size=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = "cpu"


def train(epoch):
    for e in range(epoch):
        for t, (x, y) in enumerate(mini_trains):
            x = x.to(device)
            y = y.to(device)
            model.train()

            # [N, 1, 28, 28]
            x = x.squeeze(1)
            # [N, 28, 28]
            scores = model(x)

            # loss function
            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(scores, y)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % 50 == 0:
                print(f"Epoch: {e + 1} / Iteration: {t}")
                evaluate_predictor(model)


def evaluate_predictor(model):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_data = 0

    with torch.no_grad():
        for x, y in mini_vals:
            x = x.squeeze(1)
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            predictions = scores.max(1)[1]
            acc_tensor = predictions.eq(y)
            total_accuracy += sum(acc_tensor)

            loss_function = torch.nn.CrossEntropyLoss()
            loss = loss_function(scores, y)
            total_loss += loss

            num_data += len(x)
    print(f"Val Acc: {total_accuracy / num_data} / Val Loss: {total_loss / num_data}")


train(2)
