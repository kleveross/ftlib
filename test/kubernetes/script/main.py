from __future__ import print_function

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from ftlib import BasicFTLib
from ftlib.utils.kubernetes import get_peer_set

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)

parser = argparse.ArgumentParser(
    description="Process arguments for pytorch + gossip test."
)

parser.add_argument(
    "--svc_name",
    metavar="S",
    type=str,
    default="",
    help="the name of headless service",
)

parser.add_argument(
    "--dummy_sample_num",
    metavar="D",
    type=int,
    default=1000,
    help="number of samples in dummy dataset",
)


class SyntheticData(torch.utils.data.Dataset):
    def __init__(self, generater_func, length=10):
        self._len = length
        self.generater_func = generater_func

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        x = np.random.rand() * 10.0 - 5.0
        y = self.generater_func(x)
        return np.float32(x), np.float32(y)


# define NN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.a = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.b = torch.nn.Parameter(torch.rand(1), requires_grad=True)
        self.c = torch.nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        y_predict = self.a * x * x + self.b * x + self.c
        return y_predict


class TrainingApproach:
    def __init__(self, raw_model):
        self._raw_model: nn.Module = raw_model
        self._ddp_model: nn.parallel.DistributedDataParallel = None
        self._optimizer = None
        self.need_reinit = True
        self.single_worker = False

    def _train_step(self, data, target, loss_func: nn.functional):
        output = self._ddp_model(data)
        loss = loss_func(output, target)
        print(f"loss = {loss}")
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def checkpoint(self):
        if self._ddp_model is not None:
            self._raw_model.load_state_dict(
                self._ddp_model.state_dict(), strict=False,
            )

    def train_step(self, *args, **kwargs):
        if self.need_reinit:
            if dist.is_initialized():
                # parallel mode
                print("wait for barrier")
                dist.barrier()
                print("start to broadcast")
                for p in self._raw_model.parameters():
                    dist.broadcast(p.data, 0)
                print("wrap with DDP")
                self._ddp_model = nn.parallel.DistributedDataParallel(
                    self._raw_model,
                    broadcast_buffers=False,
                    check_reduction=True,
                )
            else:
                # single worker mode
                # skip all reduce
                print("single worker mode")
                self._ddp_model = self._raw_model

            self._optimizer = optim.SGD(self._ddp_model.parameters(), lr=1e-3)
            self.need_reinit = False
        self._train_step(*args, **kwargs)


if __name__ == "__main__":

    args = parser.parse_args()

    logging.info("start!")
    logging.info("joining: {}".format(args.svc_name))

    epochs = 1

    # initialize the fault-tolerant library with consensus
    # and framework options
    ftlib = BasicFTLib(
        consensus="gossip",
        commlib="pytorch",
        consensus_init_kwargs={
            "known_addr_list": list(get_peer_set(args.svc_name))
        },
    )

    a_ground_truth = np.double(1.2)
    b_ground_truth = np.double(-3.7)
    c_ground_truth = np.double(4.9)

    target_func = (
        lambda x: a_ground_truth * x * x + b_ground_truth * x + c_ground_truth
    )

    train_loader = torch.utils.data.DataLoader(
        SyntheticData(
            lambda x: target_func(x)
            + 10.0 * (np.double(np.random.rand()) - 0.5),
            args.dummy_sample_num,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    criterion = torch.nn.MSELoss(reduction="sum")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)

    ta = TrainingApproach(model)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            if not ftlib.initialized():
                ta.need_reinit = True
                ta.checkpoint()
            ftlib.execute(ta.train_step, data, target, loss_func=criterion)
            time.sleep(5)

    logging.info("terminate!")
