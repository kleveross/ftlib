from __future__ import print_function

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from ftlib import BasicFTLib

root_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.insert(0, os.path.abspath(root_dir))


LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)

parser = argparse.ArgumentParser(
    description="Process arguments for pytorch + gossip test."
)

parser.add_argument(
    "--known-nodes",
    metavar="K",
    type=str,
    default="",
    help="hostname or ip of existing nodes, \
         separated by comma (default: None)",
)


# define dummy dataset as well as dataloader
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, length=1000):
        self._len = length

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return (torch.Tensor(np.random.rand(1, 28, 28)), np.random.randint(10))


train_loader = torch.utils.data.DataLoader(
    DummyDataset(), batch_size=8, shuffle=False
)


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


if __name__ == "__main__":

    args = parser.parse_args()
    known_addr_list = (
        args.known_nodes.split(",") if args.known_nodes != "" else []
    )

    logging.info("start!")
    logging.info("joining: {}".format(known_addr_list))

    epochs = 1

    # initialize the fault-tolerant library with consensus
    # and framework options
    ftlib = BasicFTLib(
        consensus="gossip",
        commlib="pytorch",
        consensus_init_kwargs={"known_addr_list": known_addr_list},
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # move data to device (CPU or GPU)
            data, target = data.to(device), target.to(device)
            # clear gradients
            optimizer.zero_grad()
            # forward computation
            output = model(data)
            # calculate loss
            loss = F.nll_loss(output, target)
            # backward propagation
            loss.backward()
            # call ftlib before update gradients (optimizer.step())
            if ftlib.skip_allreduce():
                logging.info("skip allreduce")
                optimizer.step()
            else:
                res = ftlib.wait_gradients_ready(model)
                if res is None:
                    optimizer.step()
        scheduler.step()

    logging.info("terminate!")
