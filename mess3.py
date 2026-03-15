"""Mess3 process: a 3-state HMM on alphabet {0,1,2}.

Parameterized by (alpha, x) with beta = (1-alpha)/2 and y = 1-2x.
Labeled transition matrices T^(z) from Shai et al. (2026), equations 22-24.
T^(z)_{s,s'} = P(emit z, transition to s' | currently in state s).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class Mess3:
    """Single ergodic Mess3 instance."""

    def __init__(self, alpha: float, x: float):
        self.alpha = alpha
        self.x = x
        self.beta = (1 - alpha) / 2
        self.y = 1 - 2 * x
        self.n_states = 3
        self.n_tokens = 3
        self.T = self._build_transition_matrices()
        # T_net[s, s'] = sum_z T^(z)[s, s'] = P(transition to s' | s)
        self.T_net = self.T.sum(axis=0)

    def _build_transition_matrices(self):
        a, b, x, y = self.alpha, self.beta, self.x, self.y
        # T[z, s, s'] = P(emit z, go to s' | in state s)
        T = np.zeros((3, 3, 3))
        T[0] = [[a*y, b*x, b*x],
                 [a*x, b*y, b*x],
                 [a*x, b*x, b*y]]
        T[1] = [[b*y, a*x, b*x],
                 [b*x, a*y, b*x],
                 [b*x, a*x, b*y]]
        T[2] = [[b*y, b*x, a*x],
                 [b*x, b*y, a*x],
                 [b*x, b*x, a*y]]
        return T

    def sample(self, length: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample a token sequence of given length from stationary distribution."""
        rng = rng or np.random.default_rng()
        tokens = np.empty(length, dtype=np.int64)
        # Stationary distribution is uniform
        state = rng.integers(0, 3)
        for t in range(length):
            # Joint distribution over (token, next_state) from current state
            probs = self.T[:, state, :].ravel()  # shape (9,) = 3 tokens * 3 states
            idx = rng.choice(9, p=probs)
            token, next_state = divmod(idx, 3)
            tokens[t] = token
            state = next_state
        return tokens

    def predictive_vectors(self, sequences: np.ndarray) -> np.ndarray:
        """Compute ground-truth predictive vectors for each context.

        Args:
            sequences: (batch, seq_len) token sequences

        Returns:
            (batch, seq_len, 3) array of predictive vectors η^(x_{1:t})
        """
        batch, seq_len = sequences.shape
        eta = np.full((batch, seq_len, 3), 1.0 / 3)  # start from stationary
        for b in range(batch):
            state = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
            for t in range(seq_len):
                z = sequences[b, t]
                state = state @ self.T[z]
                state = state / state.sum()
                eta[b, t] = state
        return eta


class Mess3Dataset(Dataset):
    """Non-ergodic dataset: multiple Mess3 instances, each sequence from one instance."""

    def __init__(self, instances: list[Mess3], seq_len: int, seqs_per_instance: int, seed: int = 42):
        self.instances = instances
        self.seq_len = seq_len
        self.seqs_per_instance = seqs_per_instance
        self.n_instances = len(instances)
        self.total = self.n_instances * seqs_per_instance

        rng = np.random.default_rng(seed)
        self.data = np.empty((self.total, seq_len), dtype=np.int64)
        self.instance_ids = np.empty(self.total, dtype=np.int64)

        for i, inst in enumerate(instances):
            for j in range(seqs_per_instance):
                idx = i * seqs_per_instance + j
                self.data[idx] = inst.sample(seq_len, rng=rng)
                self.instance_ids[idx] = i

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), self.instance_ids[idx]
