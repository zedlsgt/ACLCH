import torch
from scipy.io import loadmat, savemat
import numpy as np


def zero2eps(x):
    x[x == 0] = 1
    return x


def normalize(affinity, dim):
    affinity = affinity + torch.eye(dim).float().cuda()
    col_sum = zero2eps(torch.sum(affinity, 1)[:, None])
    out_affnty = affinity/col_sum
    return out_affnty


def gen_simples_adj(label_matrix, tau=0.6):
    label_matrix = label_matrix.float()
    a = torch.mm(label_matrix, label_matrix.t())
    adj = torch.ones(size=(label_matrix.size(0), label_matrix.size(0)))

    adj = 1 - torch.exp(tau * torch.neg(a))
    # adj.fill_diagonal_(1)
    return adj


def gen_simples_adj2(labels, tau=0.6):
    label_sim = torch.matmul(labels.float(), labels.float().T)
    l1_norm = torch.sum(labels, dim=1, keepdim=True)  # N*1矩阵
    denominator = l1_norm + l1_norm.T - label_sim
    denominator = denominator.clamp(min=1e-6)
    adj_label = label_sim/denominator
    return adj_label


def gen_normalize(A):
    D = torch.pow(A.sum(axis=1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


def set_to_zero(tensor, threshold):
    return torch.where(tensor < threshold, torch.tensor(0, dtype=tensor.dtype), tensor)
"""
#based on http://stackoverflow.com/questions/7323664/python-generator-pre-fetch
"""

import threading
import sys

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """

        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


# decorator
class background:
    def __init__(self, max_prefetch=1):
        self.max_prefetch = max_prefetch

    def __call__(self, gen):
        def bg_generator(*args, **kwargs):
            return BackgroundGenerator(gen(*args, **kwargs), max_prefetch=self.max_prefetch)

        return bg_generator
