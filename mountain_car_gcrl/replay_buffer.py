"""
Simple replay buffer for goal-conditioned RL.

Stores transitions of the form:
    (observation, action, reward, next_observation, done)

where observation = [state | goal] (already goal-augmented).

Uses a fixed-size circular buffer backed by NumPy arrays for efficiency.
"""

import numpy as np


class ReplayBuffer:
    """Fixed-capacity circular replay buffer.

    Each stored transition keeps the full goal-augmented observation so that
    the training code never has to worry about re-assembling goals.

    Args:
        obs_dim:    Dimension of the goal-augmented observation (state + goal).
        action_dim: Dimension of the action vector.
        capacity:   Maximum number of transitions to store before overwriting
                    old ones.
    """

    def __init__(self, obs_dim: int, action_dim: int, capacity: int = 500_000):
        self.capacity   = capacity
        self.obs_dim    = obs_dim
        self.action_dim = action_dim

        # Pre-allocate arrays for every field
        self._obs      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self._actions  = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards  = np.zeros((capacity, 1),          dtype=np.float32)
        self._dones    = np.zeros((capacity, 1),          dtype=np.float32)

        self._ptr  = 0   # next write position
        self._size = 0   # current number of stored transitions

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Store a single transition."""
        self._obs[self._ptr]      = obs
        self._next_obs[self._ptr] = next_obs
        self._actions[self._ptr]  = action
        self._rewards[self._ptr]  = reward
        self._dones[self._ptr]    = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> dict:
        """Sample a random mini-batch of transitions.

        Returns a dictionary of NumPy arrays, each of shape
        ``(batch_size, *dim)``.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Buffer only has {self._size} transitions, "
                f"but batch_size={batch_size} was requested."
            )
        idx = np.random.randint(0, self._size, size=batch_size)
        return {
            "obs":      self._obs[idx],
            "action":   self._actions[idx],
            "reward":   self._rewards[idx],
            "next_obs": self._next_obs[idx],
            "done":     self._dones[idx],
        }

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def ready(self) -> bool:
        """True once enough transitions have been collected to start training."""
        return self._size >= 1000
