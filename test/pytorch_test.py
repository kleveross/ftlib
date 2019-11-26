from __future__ import print_function
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np

root_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.insert(0, os.path.abspath(root_dir))

from ftlib import BasicFTLib
from ftlib.ftlib_status import *

import logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL)


# define dummy dataset as well as dataloader
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length=1000):
        self._len = length
    def __len__(self):
        return self._len
    def __getitem__(self, index):
        return (torch.Tensor(np.random.rand(1,28,28)), np.random.randint(10))
    
train_loader = torch.utils.data.DataLoader(DummyDataset(), batch_size=8, shuffle=False)

# define NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == '__main__':
    logging.info("start!")

    epochs = 1

    # initialize the fault-tolerant library with consensus and framework options
    ftlib = BasicFTLib()
    ftlib.init(consensus='shared_storage', framework='pytorch')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            time.sleep(0.5)
            loss.backward()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            if ftlib.skip_allreduce:
                logging.info("skip allreduce")
                optimizer.step()
                continue
            else:
                res = ftlib.wait_weights_ready(model)
            if res == FTAllReduceStatus.NO_NEED:
                logging.critical(
                    "cannot use average_gradient when there is no need")
                exit(2)
            if res == FTAllReduceStatus.SUCCESS:
                logging.info("average succeed")
                optimizer.step()
            if res == FTAllReduceStatus.ABORT:
                logging.info("average failed, abort")
                continue
        scheduler.step()

    logging.info("terminate!")
