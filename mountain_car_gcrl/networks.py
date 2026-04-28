"""
Neural networks for Contrastive Goal-Conditioned RL.

Two networks are defined:

1. **Actor** (policy network)
   Input : goal-augmented observation  [state | goal]  (shape: obs_dim,)
   Output: parameters of a Gaussian policy over actions, squashed through
           tanh so that actions lie in [-1, 1].

2. **ContrastiveCritic** (value / Q network)
   Instead of predicting a scalar Q-value the critic learns *representations*:
     • φ(s, a)  – encodes the current (state, action) pair
     • ψ(g)     – encodes the goal
   The Q-value estimate is their dot product:
       Q(s, a, g) = φ(s, a) · ψ(g)

   This mirrors the contrastive RL formulation in the original SGCRL paper.
   The critic is trained with an NCE (Noise Contrastive Estimation) loss:
   positives = (s_t, a_t) paired with the *actual* future state s_{t+1},
   negatives = (s_t, a_t) paired with randomly shuffled future states.

Both networks use simple MLP architectures so that the algorithm is easy
to understand and fast to run on a CPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def mlp(input_dim: int, hidden_sizes: tuple, output_dim: int) -> nn.Sequential:
    """Build a fully-connected network with ReLU activations."""
    layers = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), nn.ReLU()]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------
# Actor (policy)
# ------------------------------------------------------------------

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


class Actor(nn.Module):
    """Gaussian policy with tanh squashing.

    Outputs mean and log-std of a Normal distribution.  Actions are squashed
    through tanh so they always lie in (-1, 1).

    Args:
        obs_dim:      Dimension of the goal-augmented observation.
        action_dim:   Dimension of the action space.
        hidden_sizes: Tuple of hidden-layer widths.
    """

    def __init__(
        self,
        obs_dim:      int,
        action_dim:   int,
        hidden_sizes: tuple = (256, 256),
    ):
        super().__init__()
        self.trunk   = mlp(obs_dim, hidden_sizes[:-1], hidden_sizes[-1])
        self.mean_fc = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std_fc = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs: torch.Tensor):
        """Return (mean, log_std) of the un-squashed Gaussian."""
        h       = self.trunk(obs)
        mean    = self.mean_fc(h)
        log_std = self.log_std_fc(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and compute its log-probability.

        Returns:
            action:   Tanh-squashed action in (-1, 1), shape (B, action_dim).
            log_prob: Log-probability of the sampled action, shape (B, 1).
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Reparameterised sample
        x_t = dist.rsample()

        # Squash through tanh
        action = torch.tanh(x_t)

        # Correct log-prob for the tanh transformation (change of variables)
        # log π(a|s) = log N(x_t; μ, σ) - Σ log(1 - tanh²(x_t))
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the deterministic (mean) action – used during evaluation."""
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


# ------------------------------------------------------------------
# Contrastive critic
# ------------------------------------------------------------------

class ContrastiveCritic(nn.Module):
    """Contrastive value function.

    Learns two encoders:
      • sa_encoder: φ(s, a)  maps (state, action) → representation vector
      • g_encoder:  ψ(g)     maps goal            → representation vector

    The Q-value is estimated as the dot product:
        Q(s, a, g) = φ(s, a) · ψ(g)

    During training a batch of (s, a, g+) pairs are given where g+ is the
    actual *future* state.  The NCE loss treats the diagonal of the
    (batch × batch) dot-product matrix as positives and the off-diagonal
    entries as negatives.

    Args:
        state_dim:    Dimension of the *state* part of the observation
                      (NOT the full goal-augmented obs).
        action_dim:   Dimension of the action space.
        goal_dim:     Dimension of the goal vector.
        repr_dim:     Dimensionality of the learned representations.
        hidden_sizes: Tuple of hidden-layer widths.
        repr_norm:    If True, L2-normalise both representations before the
                      dot product.  Keeps Q-value magnitudes bounded in [-1, 1]
                      and stabilises early training.
    """

    def __init__(
        self,
        state_dim:    int,
        action_dim:   int,
        goal_dim:     int,
        repr_dim:     int   = 64,
        hidden_sizes: tuple = (256, 256),
        repr_norm:    bool  = True,
    ):
        super().__init__()
        self.repr_norm = repr_norm
        # φ(s, a)
        self.sa_encoder = mlp(state_dim + action_dim, hidden_sizes, repr_dim)
        # ψ(g)
        self.g_encoder  = mlp(goal_dim,               hidden_sizes, repr_dim)

    def encode_sa(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Encode a (state, action) pair → φ(s, a).  Shape: (B, repr_dim)."""
        h = self.sa_encoder(torch.cat([state, action], dim=-1))
        if self.repr_norm:
            h = F.normalize(h, dim=-1)   # L2 unit-sphere normalisation
        return h

    def encode_g(self, goal: torch.Tensor) -> torch.Tensor:
        """Encode a goal → ψ(g).  Shape: (B, repr_dim)."""
        h = self.g_encoder(goal)
        if self.repr_norm:
            h = F.normalize(h, dim=-1)
        return h

    def forward(
        self,
        state:  torch.Tensor,
        action: torch.Tensor,
        goal:   torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q-values as the dot product φ(s,a) · ψ(g).

        When called with batches of shape (B, *), this returns a
        (B, B) matrix of all pairwise dot products – needed for the NCE loss.
        The diagonal entry [i, i] is the Q-value for the i-th (s, a, g) triple.
        """
        sa_repr = self.encode_sa(state, action)   # (B, repr_dim)
        g_repr  = self.encode_g(goal)              # (B, repr_dim)
        # Outer dot-product:  logits[i, j] = φ(s_i, a_i) · ψ(g_j)
        logits = sa_repr @ g_repr.T                # (B, B)
        return logits

    def q_value(
        self,
        state:  torch.Tensor,
        action: torch.Tensor,
        goal:   torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar Q-value for each (s, a, g) triple.  Shape: (B, 1)."""
        sa_repr = self.encode_sa(state, action)   # (B, repr_dim)
        g_repr  = self.encode_g(goal)              # (B, repr_dim)
        # Element-wise dot product (diagonal of the full outer product)
        return (sa_repr * g_repr).sum(dim=-1, keepdim=True)  # (B, 1)
