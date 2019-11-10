import os
import sys
import time

root_dir = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.insert(0, os.path.abspath(root_dir))

from ftlib import BasicFTLib

import logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'WARNING').upper()
logging.basicConfig(level=LOGLEVEL)


def dummy_dataloader(num_bs):
    count = 0
    while count < num_bs:
        yield count
        count = count + 1


def dummy_forward():
    logging.info("dummy forward")
    time.sleep(2)


def dummy_backward():
    logging.info("dummy backward")
    time.sleep(5)


def dummy_update():
    logging.info("dummy update")
    time.sleep(0.5)


if __name__ == '__main__':
    logging.info("start!")

    epochs = 1
    dl = dummy_dataloader(10)

    # initialize the fault-tolerant library with consensus and framework options
    ftlib = BasicFTLib()
    ftlib.init(consensus='shared_storage', framework='dummy_NCCL')

    for _ in range(epochs):
        for batch in dl:
            dummy_forward()
            dummy_backward()

            if ftlib.skip_allreduce:
                logging.info("skip allreduce")
                dummy_update()
                continue
            else:
                res = ftlib.wait_weights_ready()
            if res == 'no need':
                logging.critical(
                    "cannot use average_gradient when there is no need")
                exit(2)
            if res == 'success':
                logging.info("average succeed")
                dummy_update()
            if res == 'abort':
                logging.info("average failed, abort")
                continue

    logging.info("terminate!")
