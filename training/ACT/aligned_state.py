"""GJB's seed, RNG and consumed-batch sampler; original logic preserved."""
import os
import random
from collections import deque
import numpy as np
import torch
from torch.utils.data import Sampler
def seed_everything(seed):
    # cuDNN alone is insufficient: BF16 attention/other CUDA reductions can
    # diverge after restore despite identical samples and RNG. Configure this
    # before the first CUDA matmul and reject nondeterministic implementations.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Fixed seed and recorded numerical settings support comparison; different
    # GPU libraries/machines are not promised bitwise reproducible.
    torch.backends.cudnn.deterministic = True


class StatefulBatchSampler(Sampler):
    """Checkpoint consumed batches, not batches merely prefetched by workers."""
    def __init__(self, size, batch_size, seed, prefetch_safe=False):
        if size < 1 or batch_size < 1:
            raise ValueError("Positive sample and batch counts required")
        self.size, self.batch_size = int(size), int(batch_size)
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor, self.epoch = 0, 0
        self.prefetch_safe = prefetch_safe
        self._pending = deque()
        self._committed = self._issued_state()

    def __iter__(self):
        while True:
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.cursor = 0; self.epoch += 1
            end = min(self.size, self.cursor + self.batch_size)
            batch = self.order[self.cursor:end].tolist()
            self.cursor = end
            if self.prefetch_safe:
                self._pending.append(self._issued_state())
            yield batch

    def _issued_state(self):
        return {"size": self.size, "batch_size": self.batch_size, "order": self.order,
                "cursor": self.cursor, "epoch": self.epoch, "generator": self.generator.get_state()}

    def mark_consumed(self):
        if self.prefetch_safe:
            if not self._pending:
                raise RuntimeError("No issued batch to commit")
            self._committed = self._pending.popleft()

    def state_dict(self):
        return self._committed if self.prefetch_safe else self._issued_state()

    def load_state_dict(self, state):
        if (state["size"], state["batch_size"]) != (self.size, self.batch_size):
            raise ValueError("Resume sample count/batch size changed")
        order = state["order"].cpu()
        if not torch.equal(torch.sort(order).values, torch.arange(self.size)):
            raise ValueError("Checkpoint sampler order is not a permutation")
        if not 0 <= state["cursor"] <= self.size:
            raise ValueError("Checkpoint sampler cursor invalid")
        self.order, self.cursor, self.epoch = order, state["cursor"], state["epoch"]
        self.generator.set_state(state["generator"].cpu())
        self._pending.clear()
        self._committed = self._issued_state()


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([v.cpu() for v in state["cuda"]])
