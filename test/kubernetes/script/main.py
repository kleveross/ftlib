from __future__ import print_function

import argparse
import logging
import os
import socket
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ftlib import BasicFTLib

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


def get_peer_set(svc_name):
    my_ip = socket.gethostbyname(socket.gethostname())
    temp_set = socket.getaddrinfo(svc_name, 0, proto=socket.IPPROTO_TCP)
    peer_set = {peer[-1][0] for peer in temp_set if peer[-1][0] != my_ip}
    return peer_set


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


class TrainingApproach:
    def __init__(self, raw_model):
        self._raw_model = raw_model
        self._ddp_model = nn.parallel.DistributedDataParallel(
            self._raw_model, broadcast_buffers=False, check_reduction=True,
        )
        self._optimizer = optim.SGD(self._ddp_model.parameters(), lr=1.0)
        self._future = None

    def _train_step(self, data, target, loss_func: nn.functional):
        output = self._ddp_model(data)
        loss = loss_func(output, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def train_step(self, *args, **kwargs):
        self._train_step(*args, **kwargs)

    # def train_step(self, *args, **kwargs):
    #     if self._future is not None:
    #         if not self._future.done():
    #             logging.info(
    #                 "Previous try is not completed, rebuilding ddp model"
    #             )
    #             self._ddp_model = nn.parallel.DistributedDataParallel(
    #                 model=self._raw_model,
    #                 broadcast_buffers=False,
    #                 check_reduction=True,
    #             )
    #             self._optimizer = optim.SGD(
    #                 self._ddp_model.parameters(), lr=1.0
    #             )
    #
    #     executor = ThreadPoolExecutor(max_workers=1)
    #     logging.debug("thread executor created")
    #     self._future = executor.submit(self._train_step, *args, **kwargs)
    #     logging.debug("optimizer.step submitted")
    #     start = time.time()
    #     logging.debug(f"start at: {start}")
    #     while (time.time() - start) < 10:
    #         time.sleep(0.01)
    #         logging.debug(f"e-time: {time.time() - start}")
    #         if self._future.done():
    #             logging.debug("the futures is finished")
    #             break
    #     executor.shutdown(wait=False)
    #     if not self._future.done():
    #         raise TimeoutError("step function timed-out in optimizer.step()")
    #     else:
    #         self._raw_model.load_state_dict(
    #             copy.deepcopy(self._ddp_model.state_dict()), strict=False
    #         )
    #         logging.debug("raw_model weights copied")


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

    train_loader = torch.utils.data.DataLoader(
        DummyDataset(args.dummy_sample_num),
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)

    # insert a barrier here and broadcast the weights
    ftlib.barrier()
    logging.info(f"start to broadcast model parameters from rank 0")
    for p in model.parameters():
        ftlib.broadcast(p.data, 0)
    logging.debug(model.state_dict())

    ta = TrainingApproach(model)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            ftlib.execute(ta.train_step, data, target, loss_func=F.nll_loss)
            time.sleep(1)

    logging.info("terminate!")
