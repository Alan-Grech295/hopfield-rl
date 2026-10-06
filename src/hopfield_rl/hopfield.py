import math

import torch
from torch import nn


class HopfieldNetwork(nn.Module):
    def __init__(self, dims, num_memories, temperature=1, memories=None):
        super(HopfieldNetwork, self).__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.dims = dims
        self.num_memories = num_memories
        # Create the network memories
        if memories is None:
            memories = torch.randn(num_memories, dims, dtype=torch.float64)
            memories = torch.nn.functional.normalize(memories, dim=-1)
        else:
            memories = torch.tensor(memories, dtype=torch.float64)
        self.register_buffer("memories", memories)
        self.beta = 1.0 / temperature

    def weights(self, x):
        """How well does the current state match the stored memory"""
        s = self.beta * (self.memories @ x)
        p = nn.functional.softmax(s, dim=-1)

        return p

    def update(self, x, drive):
        """Returns the state after one update iteration"""
        p = self.weights(x)
        retrieved = self.memories.T @ p
        return drive + retrieved

    def energy(self, x, drive):
        # Pull towards drive
        drive_pull = 0.5 * torch.sum((x - drive) ** 2)

        # Pull towards memories
        memory_pull = -(1 / self.beta) * (torch.logsumexp(self.beta * (self.memories @ x), dim=-1) - math.log(self.num_memories))

        return drive_pull + memory_pull

    def jacobian(self, x):
        p = self.weights(x)
        softmax_derivative = torch.diag(p) - torch.outer(p, p)
        return self.beta * self.memories.T @ softmax_derivative @ self.memories
