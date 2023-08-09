import torch
import torchtext
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IRIS import datasetIRIS
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 10).double()
        self.fc2 = nn.Linear(10, 3).double()
        self.train_losses = []
        self.train_counter = []
        #self.test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def forward(self, x):
        x.view(-1, 4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train(networkp, epoch):
  networkp.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = networkp(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      torch.save(network.state_dict(), 'test.pth')
      torch.save(optimizer.state_dict(), 'to.pth')

def test(networkp, test_loader):
  networkp.eval()
  test_loss = 0
  correct = 0
  debug = 0
  with torch.no_grad():
    for data, target in test_loader:
      # print(data)
      output = networkp(data)
      # print(output)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      # print(pred)
      correct += pred.eq(target.data.view_as(pred)).sum()
      debug+=1
  test_loss /= len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def testsimple(networkp, couple):
  networkp.eval()

  with torch.no_grad():
    data, target = couple
    print("Debug:")
    print(data)
    print("Target!!!!!!!!!!!!!!!!: ")
    print(target)
    output = networkp(data)
    print(output)
    pred = output.data.max(1, keepdim=True)[1]
    b = pred.item()
    print(b)
    plt.title(b)
    plt.imshow(data.squeeze(), cmap="gray")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    n_epochs = 800
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    train_dataset = datasetIRIS("iris/Iris.csv")
    test_dataset = datasetIRIS("iris/testset.csv")
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size_test, shuffle=True)

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                          momentum=momentum)
    test(network, test_loader)
    for epoch in range(1, n_epochs + 1):
        train(network, epoch)
        test(network, test_loader)