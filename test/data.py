import itertools

import torch


class TrickyIterator:
    # `TrickyIterator` is only for losing worker demos.
    #
    # Unlike the built-in `iter`, `TrickyIterator` is aware of the
    # change from FTLib instance. The remaining indeces from all
    # ranks will be gathered and re-distributed according to the
    # new rank and size.

    def __init__(self, epoch, ftlib, ori_data_len, shuffle=False):
        if ftlib.rank is None or ftlib.size is None:
            ftlib.build()

        assert ftlib.rank is not None
        assert ftlib.size is not None

        self.epoch = epoch
        self.ftlib = ftlib
        self.shuffle = shuffle

        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            self.total_indeces = torch.randperm(
                ori_data_len, generator=g
            ).tolist()
        else:
            self.total_indeces = list(range(ori_data_len))

        self.indeces = self.total_indeces[
            self.ftlib.rank :: self.ftlib.size  # noqa: E203
        ]

        self.cur_idx = 0

        self.rank = self.ftlib.rank
        self.size = self.ftlib.size

    def __next__(self):

        if self.rank != self.ftlib.rank or self.size != self.ftlib.size:
            indeces_remained = [
                self.total_indeces[
                    self.cur_idx * self.size + rank :: self.size  # noqa: E203
                ]
                for rank in range(self.size)
            ]
            indeces_remained = list(itertools.chain(*indeces_remained))
            print(len(indeces_remained))
            self.indeces = indeces_remained[
                self.ftlib.rank :: self.ftlib.size  # noqa: E203
            ]
            print(len(self.indeces))
            self.cur_idx = 0
            self.rank = self.ftlib.rank
            self.size = self.ftlib.size

        if self.cur_idx < len(self.indeces):
            out = self.indeces[self.cur_idx]
            self.cur_idx = self.cur_idx + 1
        else:
            raise StopIteration

        return out

    next = __next__


class TrickySampler(torch.utils.data.DistributedSampler):
    # `TrickySampler` is only for losing worker demo
    #
    # `TrickySampler` overrides the `__iter__` method of
    # `torch.utils.data.DistributedSampler` and replaces the
    # built-in `iter` with the dynamic `TrickyIterator`.

    def set_ftlib(self, ftlib):
        self.ftlib = ftlib

    def __iter__(self):
        return TrickyIterator(self.epoch, self.ftlib, len(self.dataset))
