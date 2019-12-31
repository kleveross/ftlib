from __future__ import print_function

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import TrickySampler
from torch.optim.lr_scheduler import StepLR

from ftlib import BasicFTLib
from ftlib.ftlib_status import FTAllReduceStatus

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

    # create dataset
    ds = DummyDataset()
    # create sampler
    # TrickySampler is merely a data tool for demo.
    # This sampler can only handle situation of worker
    # lost. Worker join is not considered for this sampler.
    #
    # Meanwhile, samples that has been read but not yet fed
    # the neural network are all considered as "used" and will
    # not be re-queued into the sampler after rank or size
    # changes.
    sampler = TrickySampler(ds, rank=0, num_replicas=1, shuffle=False)
    sampler.set_ftlib(ftlib)
    # create dataloader
    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        sampler=sampler,
    )

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
            time.sleep(5)
            loss.backward()
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )

            if ftlib.skip_allreduce():
                logging.info("skip allreduce")
                optimizer.step()
                continue
            else:
                res = ftlib.wait_gradients_ready(model)
            if res == FTAllReduceStatus.NO_NEED:
                logging.critical(
                    "cannot use average_gradient when there is no need"
                )
                exit(2)
            elif res == FTAllReduceStatus.SUCCESS:
                logging.info("average succeed")
                optimizer.step()
            elif res == FTAllReduceStatus.ABORT:
                logging.info("average failed, abort")
                continue
            else:
                print(type(res), res)
                logging.warning(
                    "No returned info from ftlib.wait_gradients_ready"
                )
        scheduler.step()

    logging.info("terminate!")
