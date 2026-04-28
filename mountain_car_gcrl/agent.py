"""
Contrastive Goal-Conditioned RL agent.

This module contains the ``ContrastiveGCRL`` class which ties together:

  • The **actor** (policy network) – a Gaussian policy with tanh squashing.
  • The **critic** (contrastive value network) – learns representations so
    that  Q(s, a, g) = φ(s, a) · ψ(g).
  • A **target critic** – a slowly-updated copy of the critic used to
    stabilise training (Polyak averaging).
  • **SAC-style entropy tuning** – the entropy temperature α is learnt
    automatically so that the policy maintains a target level of entropy.

Algorithm overview
------------------
At each training step the agent samples a mini-batch from the replay buffer
and performs the following updates:

1. **Critic update (NCE loss)**
   The contrastive critic is trained to distinguish *true* future states
   (positives) from randomly shuffled future states in the batch (negatives).
   Concretely, given a batch of (s, a, s') transitions augmented with a
   shared goal g:

       logits[i, j] = φ(s_i, a_i) · ψ(s'_j)   (B × B matrix)

   We want logits[i, i] to be large (positive pair) and logits[i, j≠i] to be
   small (negative pairs).  This is the standard NCE / InfoNCE loss:

       L_critic = CrossEntropy(logits, labels=eye(B))

2. **Actor update (SAC)**
   The actor maximises the Q-value of its actions minus an entropy penalty:

       L_actor = -Q(s, π(s), g) + α · log π(a|s)

   where α (the entropy temperature) is updated separately to keep
   log π ≈ -action_dim (target entropy).

3. **Alpha update (entropy temperature)**
   α is adapted so that the actual policy entropy tracks a target value:

       L_α = α · (-log π(a|s) - target_entropy)

References
----------
* Eysenbach et al. (2022). Contrastive Learning as Goal-Conditioned
  Reinforcement Learning. NeurIPS.  https://arxiv.org/abs/2206.07568
* Haarnoja et al. (2018). Soft Actor-Critic.  https://arxiv.org/abs/1801.01290
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from mountain_car_gcrl.networks import Actor, ContrastiveCritic


class ContrastiveGCRL:
    """Goal-conditioned RL agent using a contrastive critic and SAC actor.

    Args:
        state_dim:      Dimension of the environment state (e.g. 2 for
                        MountainCarContinuous).
        action_dim:     Dimension of the action space (e.g. 1).
        goal_dim:       Dimension of the goal vector (e.g. 2).
        hidden_sizes:   Hidden-layer widths for both actor and critic.
        repr_dim:       Dimensionality of the contrastive representations.
        repr_norm:      If True, L2-normalise representations before the dot
                        product, keeping Q-values in [-1, 1] and stabilising
                        training (default: True).
        actor_lr:       Learning rate for the actor.
        critic_lr:      Learning rate for the critic.
        alpha_lr:       Learning rate for the entropy temperature α.
        tau:            Polyak averaging coefficient for target critic update.
                        target_params ← (1 - τ) * target + τ * current
        discount:       Discount factor γ (not used in MC contrastive loss,
                        kept for potential TD extension).
        target_entropy: Target entropy for the SAC temperature tuning.
                        Defaults to -action_dim (one nat per action dimension).
        grad_clip:      Maximum gradient norm (gradient clipping) applied to
                        both actor and critic before each optimiser step.
                        Prevents exploding gradients in early training.
        device:         Torch device ("cpu" or "cuda").
    """

    def __init__(
        self,
        state_dim:      int,
        action_dim:     int,
        goal_dim:       int,
        hidden_sizes:   tuple = (256, 256),
        repr_dim:       int   = 64,
        repr_norm:      bool  = True,
        actor_lr:       float = 3e-4,
        critic_lr:      float = 3e-4,
        alpha_lr:       float = 3e-4,
        tau:            float = 0.005,
        discount:       float = 0.99,
        target_entropy: float | None = None,
        grad_clip:      float        = 1.0,
        device:         str          = "cpu",
    ):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.goal_dim   = goal_dim
        self.obs_dim    = state_dim + goal_dim   # full goal-augmented obs dim
        self.tau        = tau
        self.discount   = discount
        self.grad_clip  = grad_clip
        self.device     = torch.device(device)

        # ------------------------------------------------------------------ #
        # Networks
        # ------------------------------------------------------------------ #
        self.actor = Actor(
            obs_dim      = self.obs_dim,
            action_dim   = action_dim,
            hidden_sizes = hidden_sizes,
        ).to(self.device)

        self.critic = ContrastiveCritic(
            state_dim    = state_dim,
            action_dim   = action_dim,
            goal_dim     = goal_dim,
            repr_dim     = repr_dim,
            hidden_sizes = hidden_sizes,
            repr_norm    = repr_norm,
        ).to(self.device)

        # Target critic is a slowly-updated copy (Polyak averaging)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------ #
        # Optimisers
        # ------------------------------------------------------------------ #
        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ------------------------------------------------------------------ #
        # Entropy temperature α (learnt automatically)
        # ------------------------------------------------------------------ #
        self.target_entropy = (
            -float(action_dim) if target_entropy is None else target_entropy
        )
        # We optimise log α for numerical stability (α = exp(log_alpha) ≥ 0)
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

    # ---------------------------------------------------------------------- #
    # Action selection
    # ---------------------------------------------------------------------- #

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy temperature (always ≥ 0)."""
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Choose an action given a goal-augmented observation.

        Args:
            obs:      NumPy array of shape (obs_dim,).
            evaluate: If True, use the deterministic (mean) action instead
                      of sampling.  Used during evaluation rollouts.

        Returns:
            action: NumPy array of shape (action_dim,) in [-1, 1].
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                action = self.actor.deterministic_action(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.squeeze(0).cpu().numpy()

    # ---------------------------------------------------------------------- #
    # Training step
    # ---------------------------------------------------------------------- #

    def update(self, batch: dict) -> dict:
        """Perform one gradient update using a sampled mini-batch.

        Args:
            batch: Dictionary from ``ReplayBuffer.sample`` with keys:
                   "obs", "action", "reward", "next_obs", "done".
                   Each value is a NumPy array of shape (B, *dim).

        Returns:
            metrics: Dictionary of scalar training metrics for logging.
        """
        # Convert batch to tensors
        obs      = torch.tensor(batch["obs"],      dtype=torch.float32, device=self.device)
        action   = torch.tensor(batch["action"],   dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)

        # Split goal-augmented observations into state and goal parts
        state      = obs[:,      :self.state_dim]   # (B, state_dim)
        goal       = obs[:,      self.state_dim:]   # (B, goal_dim)
        next_state = next_obs[:, :self.state_dim]   # (B, state_dim)

        # ------------------------------------------------------------------ #
        # 1. Critic update – NCE contrastive loss
        # ------------------------------------------------------------------ #
        # Treat the next_state as the "positive" goal for each transition.
        # The critic should predict that (s_i, a_i) → s'_i (high score on
        # the diagonal) and that (s_i, a_i) → s'_j≠i is unlikely (low score
        # off-diagonal).
        logits = self.critic(state, action, next_state)   # (B, B)

        batch_size = logits.shape[0]
        labels     = torch.arange(batch_size, device=self.device)   # [0, 1, …, B-1]

        # Symmetric NCE loss (both row-wise and column-wise)
        critic_loss = (
            F.cross_entropy(logits,   labels)   # rows  = (s,a) queries,  cols = goals
            + F.cross_entropy(logits.T, labels) # rows  = goals,           cols = (s,a)
        ) / 2.0

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # Accuracy: how often is the diagonal the highest-scoring column?
        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == labels).float().mean()

        # ------------------------------------------------------------------ #
        # 2. Actor update – maximise Q(s, π(s), g) minus entropy penalty
        # ------------------------------------------------------------------ #
        new_action, log_prob = self.actor.sample(obs)   # (B, 1), (B, 1)

        # Q-value for the actor's chosen action (scalar per sample)
        q_value = self.critic.q_value(state, new_action, goal)   # (B, 1)

        # Actor loss: minimise  α·log_π - Q   (equivalent to maximising Q - α·H)
        actor_loss = (self.alpha.detach() * log_prob - q_value).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # ------------------------------------------------------------------ #
        # 3. Alpha update – keep entropy ≈ target_entropy
        # ------------------------------------------------------------------ #
        # L_α = α · (-log π - target_entropy)
        # When -log π > target_entropy the policy is too random → increase α.
        # When -log π < target_entropy the policy is too deterministic → decrease α.
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ------------------------------------------------------------------ #
        # 4. Soft update of the target critic (Polyak averaging)
        # ------------------------------------------------------------------ #
        self._polyak_update(self.critic, self.target_critic)

        return {
            "critic_loss":  critic_loss.item(),
            "actor_loss":   actor_loss.item(),
            "alpha_loss":   alpha_loss.item(),
            "alpha":        self.alpha.item(),
            "critic_acc":   accuracy.item(),
            "mean_log_prob": log_prob.mean().item(),
        }

    # ---------------------------------------------------------------------- #
    # Internals
    # ---------------------------------------------------------------------- #

    def _polyak_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        """Update target network parameters: θ_target ← (1-τ)θ_target + τθ_source."""
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.mul_(1.0 - self.tau)
            tgt_param.data.add_(self.tau * src_param.data)

    # ---------------------------------------------------------------------- #
    # Checkpointing
    # ---------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Save all network weights to a file."""
        torch.save(
            {
                "actor":          self.actor.state_dict(),
                "critic":         self.critic.state_dict(),
                "target_critic":  self.target_critic.state_dict(),
                "log_alpha":      self.log_alpha.detach().cpu(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load network weights previously saved with ``save``."""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.target_critic.load_state_dict(ckpt["target_critic"])
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"].to(self.device))
