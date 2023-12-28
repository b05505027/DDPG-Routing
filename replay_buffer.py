import numpy as np
from typing import Dict

class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, s_dim: int, a_dim: int, max_size: int, batch_size: int = 32):
        
        """Initializate."""
        self.s_buffer = np.zeros([max_size, s_dim], dtype=np.float32)
        self.next_s_buffer = np.zeros([max_size, s_dim], dtype=np.float32)
        self.a_buffer = np.zeros([max_size, a_dim], dtype=np.float32)
        self.r_buffer = np.zeros([max_size], dtype=np.float32)
        self.done_buffer = np.zeros([max_size], dtype=np.float32)
        self.is_buffer = np.zeros([max_size], dtype=np.float32)
        self.max_size, self.batch_size = max_size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        s: np.ndarray,
        a: np.ndarray, 
        r: float, 
        next_s: np.ndarray, 
        is_ratio: float,
        done: bool,
    ):
        """Store the transition in buffer."""
        self.s_buffer[self.ptr] = s
        self.next_s_buffer[self.ptr] = next_s
        self.a_buffer[self.ptr] = a
        self.r_buffer[self.ptr] = r
        self.done_buffer[self.ptr] = done
        self.is_buffer[self.ptr] = is_ratio
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        """Randomly sample a batch of experiences from memory."""
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(s=self.s_buffer[idxs],
                    next_s=self.next_s_buffer[idxs],
                    a=self.a_buffer[idxs],
                    r=self.r_buffer[idxs],
                    is_ratio=self.is_buffer[idxs],
                    done=self.done_buffer[idxs])

    def __len__(self) -> int:
        return self.size
