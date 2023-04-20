from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from torch.utils.data import DataLoader, random_split

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    train the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate average loss and accuracy for the training set
    training_loss = train_loss / len(train_loader.dataset)
    training_acc = 100.0 * correct / len(train_loader.dataset)
    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the testing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Calculate average loss and accuracy for the test set
    testing_loss = test_loss / len(test_loader.dataset)
    testing_acc = 100.0 * correct / len(test_loader.dataset)
    return testing_acc, testing_loss


def plot(epochs, performance, title, run=None):
    """
    plot the model performance
    :param epochs: recorded epochs
    :param performance: recorded performance
    :return:
    """
    plt.plot(epochs, performance)
    plt.xlabel('Epoch')
    plt.ylabel('Percentage')

    if run is not None:
        title += f" (Run {run})" if run != 'mean' else " (Mean)"

    plt.title(title)
    plt.show()


def run(config, run_number):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    # use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(config.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Set DataLoader arguments based on whether CUDA is available
    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Load and preprocess the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    # Initialize DataLoaders for training and testing
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Initialize the model, optimizer, and learning rate scheduler
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epochs = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        test_acc, test_loss = test(model, device, test_loader)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        scheduler.step()
        epochs.append(epoch)

    plot(epochs, training_accuracies, 'Training Accuracy', run_number)
    plot(epochs, training_loss, 'Training Loss', run_number)
    plot(epochs, training_accuracies, 'Testing Accuracy', run_number)
    plot(epochs, testing_loss, 'Testing Loss', run_number)

    with open(f'results_{config.seed}.txt', 'w') as f:
        for epoch, train_acc, train_loss, test_acc, test_loss in zip(epochs, training_accuracies, training_loss,
                                                                     testing_accuracies, testing_loss):
            f.write(f"{epoch} {train_acc} {train_loss} {test_acc} {test_loss}\n")

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def plot_mean():
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    results = {}

    for seed1 in seeds:
        with open(f'results_{seed1}.txt', 'r') as f:
            lines = f.readlines()

        for line in lines:
            epoch, train_acc, train_loss, test_acc, test_loss = [float(x) for x in line.strip().split()]
            if epoch not in results:
                results[epoch] = {'train_accuracies': [], 'train_losses': [], 'test_accuracies': [], 'test_losses': []}

            results[epoch]['train_accuracies'].append(train_acc)
            results[epoch]['train_losses'].append(train_loss)
            results[epoch]['test_accuracies'].append(test_acc)
            results[epoch]['test_losses'].append(test_loss)

    epochs = list(results.keys())
    mean_train_accuracies = [np.mean(results[epoch]['train_accuracies']) for epoch in epochs]
    mean_train_losses = [np.mean(results[epoch]['train_losses']) for epoch in epochs]
    mean_test_accuracies = [np.mean(results[epoch]['test_accuracies']) for epoch in epochs]
    mean_test_losses = [np.mean(results[epoch]['test_losses']) for epoch in epochs]
    plot(epochs, mean_train_accuracies, 'Mean Training Accuracy', 'mean')
    plot(epochs, mean_train_losses, 'Mean Training Loss', 'mean')
    plot(epochs, mean_test_accuracies, 'Mean Testing Accuracy', 'mean')
    plot(epochs, mean_test_losses, 'Mean Testing Loss', 'mean')


def run_with_seed(config, seed, run_number):
    config.seed = seed
    run(config, run_number)


if __name__ == '__main__':
    args = read_args()

    # load training settings
    config = load_config(args)

    # Create a list of processes
    processes = []
    seeds = config.seeds

    # Initialize the processes with the specified random seeds
    for idx, seed in enumerate(seeds):
        process = mp.Process(target=run_with_seed, args=(config, seed, idx + 1))
        processes.append(process)

    # Start the processes
    for process in processes:
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # plot the mean results
    plot_mean()
