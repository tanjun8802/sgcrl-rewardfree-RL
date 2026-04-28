# Goal-Conditioned RL on MountainCarContinuous-v0

This folder contains a **self-contained, easy-to-read** implementation of the
contrastive goal-conditioned RL algorithm from the parent repository, adapted
to run on Gymnasium's `MountainCarContinuous-v0` environment.

No MuJoCo, no ACME, no JAX, no distributed setup required — just
**PyTorch + Gymnasium**.

---

## Algorithm Overview

### What is Goal-Conditioned RL?

In standard RL the agent tries to maximise a reward signal defined by the
environment.  In **goal-conditioned RL (GCRL)** the agent additionally receives
a *goal* at the start of each episode and must learn a policy that reaches
arbitrary goals, not just a single hardcoded objective.

The observation fed to the agent is always:

```
obs = [state | goal]          e.g. [position, velocity, goal_pos, goal_vel]
```

### Contrastive Value Function

Instead of learning a scalar Q-value Q(s, a, g) the agent learns two
*representation encoders*:

| Encoder | Input | Output |
|---------|-------|--------|
| `φ(s, a)` | current state + action | representation vector (repr_dim,) |
| `ψ(g)` | goal | representation vector (repr_dim,) |

The Q-value is their **dot product**:

```
Q(s, a, g) = φ(s, a) · ψ(g)
```

A large dot product means "this (state, action) pair is likely to lead to goal g".

### NCE Contrastive Loss

Given a mini-batch of *B* transitions `(s_i, a_i, s'_i)` (where `s'_i` is the
next state, treated as the "achieved goal"), the critic computes a `B × B`
matrix of dot products:

```
logits[i, j] = φ(s_i, a_i) · ψ(s'_j)
```

- The **diagonal** entries `logits[i, i]` are **positive pairs** — the
  (state, action) should score highly against its own future state.
- The **off-diagonal** entries are **negative pairs** — random mismatches.

This is exactly a cross-entropy classification problem where the label for
row `i` is `i`:

```
L_critic = CrossEntropy(logits, labels=I)
```

### SAC Actor Update

The actor is a Gaussian policy with tanh squashing (outputs actions in [-1, 1]).
It is trained to maximise the Q-value of its own actions, subject to an entropy
regularisation term (Soft Actor-Critic):

```
L_actor = α · log π(a|s) - Q(s, π(s), g)
         ^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^
         entropy penalty     value of action
```

The temperature `α` is learnt automatically to keep the policy's entropy
near a target value (`-action_dim` nats by default).

---

## File Structure

```
mountain_car_gcrl/
├── __init__.py        – package marker
├── env_wrapper.py     – GoalConditionedMountainCar (gym.Wrapper)
├── replay_buffer.py   – Simple circular replay buffer (NumPy)
├── networks.py        – Actor & ContrastiveCritic (PyTorch)
├── agent.py           – ContrastiveGCRL: critic + actor + alpha updates
└── train.py           – Self-contained training loop with CLI
```

Each file is thoroughly documented with docstrings and inline comments
explaining *why* each design choice was made, not just *what* it does.

---

## Quick Start

### 1. Install dependencies

```bash
pip install torch gymnasium
```

That's it.  No MuJoCo binaries, no JAX, no special conda environment.

### 2. Train the agent

```bash
# From the repository root:
python -m mountain_car_gcrl.train
```

Training runs for **300 000 environment steps** (~5–10 minutes on a laptop
CPU).  You'll see live console output and a CSV log in `runs/`.

### 3. Evaluate a saved checkpoint

```bash
python -m mountain_car_gcrl.train --eval_only --checkpoint runs/best_actor.pt
```

### 4. Watch the agent

```bash
python -m mountain_car_gcrl.train \
    --eval_only \
    --checkpoint runs/best_actor.pt \
    --render
```

---

## Key Hyperparameters

| Flag | Default | Meaning |
|------|---------|---------|
| `--total_steps` | 300 000 | Total environment interaction steps |
| `--batch_size` | 256 | Mini-batch size |
| `--warmup_steps` | 5 000 | Random steps before first gradient update |
| `--eval_freq` | 5 000 | Evaluate every N steps |
| `--seed` | 42 | Random seed |
| `--device` | cpu | `cpu` or `cuda` |

---

## Understanding the Output

Each log line looks like:

```
[step  10000] [train]  episodes=12  critic_loss=0.8234  actor_loss=-0.1234  alpha=0.2345  critic_acc=0.1523  mean_reward=0.0000  steps/s=312.4
[step  10000] [eval]   eval/success_rate=0.0000  eval/mean_reward=0.0000  eval/mean_ep_length=999.0
```

| Metric | Meaning |
|--------|---------|
| `critic_loss` | NCE cross-entropy loss.  Should decrease towards ~0 |
| `critic_acc` | Fraction of batches where diagonal score is highest (0–1).  Should rise towards 1 |
| `actor_loss` | SAC actor objective (lower is not necessarily better) |
| `alpha` | Entropy temperature.  Adapts automatically |
| `eval/success_rate` | Fraction of evaluation episodes where car reached the goal |

---

## How This Relates to the Parent Repository

| Parent (SGCRL) | This implementation |
|----------------|---------------------|
| JAX / Haiku networks | PyTorch `nn.Module` |
| ACME distributed actors / learners | Single-process loop |
| Reverb replay buffer | Simple NumPy circular buffer |
| Sawyer robot / PointSpiral environments | `MountainCarContinuous-v0` |
| Contrastive NCE / CPC loss | Same NCE loss, symmetric version |
| Adaptive entropy (SAC) | Same |
| Polyak target critic | Same |

The core algorithm is identical; only the infrastructure is simplified.
