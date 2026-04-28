# Goal-Conditioned RL on MountainCarContinuous-v0

A **self-contained, beginner-friendly** implementation of the Contrastive
Goal-Conditioned RL algorithm.  Only **PyTorch + Gymnasium** required — no
MuJoCo, no JAX, no distributed setup.

```bash
pip install torch gymnasium
python -m mountain_car_gcrl.train
```

---

## Table of Contents

1. [The Environment — What Is MountainCar?](#1-the-environment--what-is-mountaincar)
2. [What Is Goal-Conditioned RL?](#2-what-is-goal-conditioned-rl)
3. [The Big Idea: Contrastive Value Functions](#3-the-big-idea-contrastive-value-functions)
4. [What Does the Embedding Vector Actually Mean?](#4-what-does-the-embedding-vector-actually-mean)
5. [The Actor: How Does the Agent Decide What to Do?](#5-the-actor-how-does-the-agent-decide-what-to-do)
6. [The Training Loop: Putting It All Together](#6-the-training-loop-putting-it-all-together)
7. [File-by-File Walkthrough](#7-file-by-file-walkthrough)
8. [Quick Start](#8-quick-start)
9. [Key Hyperparameters](#9-key-hyperparameters)
10. [Reading the Training Output](#10-reading-the-training-output)
11. [How This Relates to the Parent Repository](#11-how-this-relates-to-the-parent-repository)

---

## 1. The Environment — What Is MountainCar?

Imagine a small car stuck in a valley between two hills.  The car's engine is
too weak to drive straight up, so it must build momentum by rocking back and
forth.

```
         ____
        /    \         ← FLAG (goal position ≈ +0.45)
       /      \
______/        \______
        🚗            ← car starts somewhere in the valley
```

At every moment the environment gives the agent two numbers:

| Variable | Range | Meaning |
|----------|-------|---------|
| `position` | −1.2 to +0.6 | Where the car is along the hill (left = negative, right = positive) |
| `velocity` | −0.07 to +0.07 | How fast and in which direction it is moving |

The agent controls one number:

| Variable | Range | Meaning |
|----------|-------|---------|
| `force` | −1.0 to +1.0 | How hard to push left (−1) or right (+1) |

The task is to reach the flag on the right hill.  In the standard environment
this earns a +100 reward; in our goal-conditioned version the reward is simply
**+1 when the car comes within 5 cm and 0.05 m/s of the goal**, and **0
otherwise**.

---

## 2. What Is Goal-Conditioned RL?

### Standard RL recap

In standard RL there is one fixed objective baked into the reward function.
You train an agent to reach the flag — and that is the only task it can ever
do.

### Adding goals

In **Goal-Conditioned RL (GCRL)** the agent receives a *goal* alongside its
state at every timestep.  The observation becomes:

```
obs = [state | goal]
    = [position, velocity,  goal_position, goal_velocity]
      └──────────────────┘  └─────────────────────────────┘
           what is now              what we want to be
```

Now the same neural network can be asked "given you are here, go *there*"
for any "there".  During training we expose the agent to many different goals
so it generalises.  During evaluation we fix the goal to the top of the hill.

### Why is this useful?

- A single policy learns to reach *any* reachable state, not just one.
- You can change the goal without retraining.
- It is inherently **reward-free**: the only signal is whether you reached the
  goal, which you know without human-designed rewards.

---

## 3. The Big Idea: Contrastive Value Functions

### The standard Q-function

Normally a Q-function `Q(s, a, g)` is a single neural network that takes the
state, action, and goal and outputs a scalar: *"how good is this action for
reaching this goal?"*

The problem is that you need many real environment interactions to learn this
Q-value accurately, especially for distant goals.

### A smarter approach: learn two embeddings

Instead of learning one scalar, we learn **two vector-valued encoders**:

```
φ(s, a)  →  a vector of length 64    (the "state-action fingerprint")
ψ(g)     →  a vector of length 64    (the "goal fingerprint")
```

The Q-value is then just their **dot product** (cosine similarity, since both
are L2-normalised):

```
Q(s, a, g)  =  φ(s, a) · ψ(g)
```

If the two vectors point in roughly the same direction → high score → the
agent is likely to reach the goal.  If they point in different directions →
low score → the action probably won't lead there.

### How are the encoders trained? — The NCE Loss

We train the encoders using **Noise Contrastive Estimation (NCE)**, the same
idea behind word2vec and contrastive image learning.

Given a mini-batch of 256 experience transitions
`(s₁,a₁,s'₁), (s₂,a₂,s'₂), … (s₂₅₆,a₂₅₆,s'₂₅₆)`:

- Each next-state `s'ᵢ` is the **actual future** reached by `(sᵢ, aᵢ)` — a
  **positive** pair.
- Every other next-state `s'ⱼ ≠ s'ᵢ` is a **negative** (a random mismatch).

We build a 256 × 256 score matrix:

```
              goal embeddings  ψ(s'₁)  ψ(s'₂)  ψ(s'₃)  …
               ┌─────────────────────────────────────────┐
φ(s₁,a₁)  →  │  HIGH    low    low   …  │   ← row 1
φ(s₂,a₂)  →  │   low   HIGH   low   …  │   ← row 2
φ(s₃,a₃)  →  │   low    low  HIGH   …  │   ← row 3
    …         │   …      …    …    HIGH │
               └─────────────────────────────────────────┘
```

We want the **diagonal** to be the highest score in each row.  This is
identical to a standard cross-entropy classification problem where the correct
"class" for row `i` is column `i`:

```python
labels = [0, 1, 2, 3, …, 255]
loss   = CrossEntropy(score_matrix, labels)
```

The encoders are pushed to make each `(s,a)` fingerprint point toward its
*own* future state and away from everyone else's.

---

## 4. What Does the Embedding Vector Actually Mean?

This is the key question.  Here is the intuition:

### φ(s, a) — the state-action embedding

> *"If the car is at position p with velocity v and pushes with force f, where
> in the space of reachable futures does this trajectory end up?"*

After training, `φ(s, a)` is a **direction in 64-dimensional space that
encodes the probable future trajectory** of taking action `a` from state `s`.

Think of it as a fingerprint of "where this move is likely to take you".
Concretely for MountainCar:

- From the **bottom of the valley pushing right**: the car gains speed to the
  right → φ points toward "high-position, high-rightward-velocity" futures.
- From the **bottom of the valley pushing left**: φ points toward
  "moving left, building negative momentum" futures.
- From **near the top pushing right**: φ should point very strongly toward
  the top-right corner of the state space, close to the goal embedding.

The vector does **not** directly encode raw position/velocity numbers.  It
encodes something richer: *the likelihood distribution over which future states
this trajectory visits*, compressed into 64 dimensions.

### ψ(g) — the goal embedding

> *"What kind of trajectory would end here?"*

`ψ(g)` is the direction in the same 64-dimensional space that represents the
goal state.  Two goals that require similar trajectories to reach will have
similar `ψ` vectors, even if their raw coordinates differ.

### The dot product as a "similarity score"

With L2 normalisation both vectors lie on a unit sphere.  Their dot product is
the **cosine of the angle between them**:

```
cos(θ) = φ(s,a) · ψ(g)

   θ ≈ 0°   (same direction)  →  cos ≈ +1.0  →  very likely to reach g
   θ ≈ 90°  (perpendicular)   →  cos ≈  0.0  →  no information about g
   θ ≈ 180° (opposite)        →  cos ≈ -1.0  →  definitely not heading to g
```

### A concrete example from MountainCar

```
Scenario: car is at position −0.5 (middle of valley), velocity = 0.

Action A: push RIGHT (+1.0)
  φ(s, A) ≈ [...points toward "right hill" region of embedding space...]
  ψ(goal at +0.45) ≈ [...points toward "right hill" region...]
  dot product ≈ +0.8   →   Q ≈ 0.8   →   "good move toward the goal"

Action B: push LEFT (−1.0)
  φ(s, B) ≈ [...points toward "left hill" region of embedding space...]
  ψ(goal at +0.45) ≈ [...still points toward "right hill"...]
  dot product ≈ −0.3   →   Q ≈ −0.3  →   "bad move for this goal"
```

This is why the representation is so powerful: the network does not need to
learn a separate Q-function for every goal.  Once it has learnt the geometry
of the embedding space, it can evaluate *any* goal for *any* state-action pair
just by computing a dot product.

---

## 5. The Actor: How Does the Agent Decide What to Do?

The **actor** is a separate neural network (the policy `π`).  It takes the
full goal-augmented observation `[state | goal]` and outputs a *distribution*
over actions.

```
obs = [position, velocity, goal_pos, goal_vel]
         ↓
     Actor network (2 × 256 hidden units)
         ↓
     (mean μ, std σ) of a Gaussian distribution
         ↓
     sample x ~ N(μ, σ)
         ↓
     action = tanh(x)   ← squashed into (−1, 1)
```

Why sample instead of always picking the mean?  **Exploration**.  Early in
training the agent needs to try many things to discover what works.  Sampling
adds noise that helps visit new states.  During evaluation we switch to the
deterministic mean (`tanh(μ)`) for the best performance.

### Entropy regularisation (SAC)

The actor is trained with a **Soft Actor-Critic (SAC)** objective that
balances two competing goals:

```
minimise:   α · log π(a|s)   −   Q(s, a, g)
            ──────────────       ───────────
            "don't be too         "choose
             deterministic"        good actions"
```

- `Q(s, a, g)` = the dot product from the critic → "how good is this action?"
- `log π(a|s)` = log-probability of the sampled action → negative entropy
- `α` = a scalar temperature that balances the two

When `α` is large the actor is pushed to spread its probability mass (explore
more).  When `α` is small it focuses on the best-known action (exploit).

### Auto-tuning α

We do not manually set `α`.  Instead `α` is trained to keep the policy entropy
at a target value (−1 nat for 1 action dimension).  If the policy becomes too
deterministic, `α` increases to force more exploration.  If it is too random,
`α` decreases.  This self-regulation happens automatically during training.

---

## 6. The Training Loop: Putting It All Together

Here is the full picture of how data flows every training step:

```
┌─────────────────────────────────────────────────────────────────┐
│                         ENVIRONMENT                             │
│  MountainCarContinuous-v0  wrapped with a goal                  │
│                                                                 │
│  obs = [position, velocity, goal_pos, goal_vel]                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ obs
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                          ACTOR                                 │
│  Takes obs → outputs action (force in [−1, 1])                 │
│  Samples from Gaussian for exploration during training         │
└────────────────────────┬───────────────────────────────────────┘
                         │ action
                         ▼
              Environment executes action
              Returns (next_obs, reward, done)
                         │
                         ▼
┌───────────────────────────────────────────────────────────────┐
│                      REPLAY BUFFER                            │
│  Stores (obs, action, reward, next_obs, done) tuples          │
│  Capacity: 500 000 transitions                                │
│  Random mini-batch of 256 drawn every training step           │
└───────────────────────┬───────────────────────────────────────┘
                        │ mini-batch of 256
                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     THREE GRADIENT UPDATES                           │
│                                                                      │
│  Step 1 — Critic (NCE loss)                                          │
│    Compute 256×256 score matrix of φ(s,a) · ψ(next_state)           │
│    Push diagonal scores up, off-diagonal scores down                 │
│    → encoders learn the geometry of reachable futures                │
│                                                                      │
│  Step 2 — Actor (SAC loss)                                           │
│    Sample new action from policy for each obs in batch               │
│    Compute Q = φ(s, new_action) · ψ(goal) using updated critic       │
│    Maximise Q − α·log_prob                                           │
│    → policy learns to pick actions that lead toward the goal          │
│                                                                      │
│  Step 3 — Alpha update                                               │
│    Adjust α so that policy entropy ≈ −1 nat                          │
│    → keeps a good exploration/exploitation balance                    │
│                                                                      │
│  Step 4 — Polyak update (no gradient)                                │
│    target_critic ← 0.995·target_critic + 0.005·critic               │
│    → stable Q-value estimates during actor updates                   │
└──────────────────────────────────────────────────────────────────────┘
```

This cycle repeats for every environment step until `--total_steps` is
reached.  The first `--warmup_steps` (5 000) use purely random actions to
fill the replay buffer before any gradients are computed.

---

## 7. File-by-File Walkthrough

### `env_wrapper.py` — The Goal-Conditioned Environment

**What it does:** Wraps Gymnasium's `MountainCarContinuous-v0` so the agent
always receives a goal alongside the current state.

```python
obs = [position, velocity, goal_position, goal_velocity]
       ───────────────────  ────────────────────────────
          current state (2)       desired state (2)
```

Key design decisions:
- **Sparse reward**: +1 only when the car is within 5 cm and 0.05 m/s of the
  goal.  No shaped reward.  This is intentional — the contrastive critic
  provides its own implicit reward signal via the dot product.
- **Random goals during training**: at each episode reset a new goal position
  in [0.1, 0.6] is sampled uniformly.  This is crucial — the agent only
  generalises to new goals if it has seen many during training.
- **Fixed goal during evaluation**: always targets position +0.45 (the flag)
  with zero velocity, so results are comparable.

### `replay_buffer.py` — Experience Memory

**What it does:** A first-in-first-out circular buffer that stores past
transitions so we can train on them repeatedly.

Why not just use the most recent experience?
- Neural networks train better on shuffled, independent mini-batches.
- Storing millions of past experiences makes each gradient update more
  informative and stable.

```
Buffer of 500 000 slots:
[ t=1 | t=2 | t=3 | … | t=500000 | ← overwrite from t=1 ]
Each slot = (obs, action, reward, next_obs, done)
```

### `networks.py` — The Two Neural Networks

**`Actor`** — the policy network:
- Input: `[position, velocity, goal_pos, goal_vel]` (4 numbers)
- Two hidden layers of 256 neurons each with ReLU activations
- Output: mean `μ` and log-std `log σ` of a Gaussian (each 1 number)
- Final action = `tanh(sample from N(μ, σ))` → always in (−1, 1)

**`ContrastiveCritic`** — the value network:
- **`sa_encoder`**: Input: `[position, velocity, force]` (3 numbers) → 64-dim vector
- **`g_encoder`**: Input: `[goal_pos, goal_vel]` (2 numbers) → 64-dim vector
- Both output vectors are L2-normalised to the unit sphere
- Q-value = dot product of the two vectors (between −1 and +1)

Note that the critic takes the *raw state* and *raw action*, **not** the
goal-augmented observation.  It encodes them separately with different networks.

### `agent.py` — The Learning Algorithm

Ties everything together.  The `update(batch)` method performs all three
gradient steps described in Section 6.  `select_action(obs)` queries the actor
to pick the next move.

### `train.py` — The Training Script

A complete training loop in a single file.  Run it directly:
```bash
python -m mountain_car_gcrl.train
```
It handles:
- Creating the environment, agent, and replay buffer
- The collect → store → sample → update cycle
- Periodic evaluation and console logging
- Saving the best checkpoint by success rate

---

## 8. Quick Start

### Install

```bash
pip install torch gymnasium
```

### Train from scratch

```bash
# From the repository root
python -m mountain_car_gcrl.train
```

Default: 300 000 steps (~5–10 minutes on a laptop CPU).  You will see live
output every 5 000 steps and a `runs/training_log.csv` you can plot.

### Evaluate a saved agent

```bash
python -m mountain_car_gcrl.train --eval_only --checkpoint runs/best_actor.pt
```

### Watch the agent drive

```bash
python -m mountain_car_gcrl.train \
    --eval_only \
    --checkpoint runs/best_actor.pt \
    --render
```

---

## 9. Key Hyperparameters

| Flag | Default | What to change it for |
|------|---------|----------------------|
| `--total_steps` | 300 000 | Increase (e.g. 1 000 000) if success rate is still rising |
| `--batch_size` | 256 | Smaller = noisier gradients; larger = slower but smoother |
| `--warmup_steps` | 5 000 | More = more random exploration before first update |
| `--updates_per_step` | 1 | Increase to train more aggressively on existing data |
| `--eval_freq` | 5 000 | How often (in steps) to run evaluation episodes |
| `--eval_episodes` | 10 | More episodes = more reliable success rate estimate |
| `--seed` | 42 | Change to check variance across runs |
| `--device` | cpu | Use `cuda` if a GPU is available |

---

## 10. Reading the Training Output

Example output:

```
[step   5000] [train] episodes=5   critic_loss=4.70  actor_loss=-0.39  alpha=0.58  critic_acc=0.60  mean_reward=0.00  steps/s=115
[step   5000] [eval]  eval/success_rate=0.00  eval/mean_reward=0.00  eval/mean_ep_length=999.0
```

### What each number means

| Metric | Good range | Meaning |
|--------|-----------|---------|
| `critic_loss` | Decreasing from ~5 to ~1 | NCE cross-entropy loss.  Lower means the embeddings are better separated. |
| `critic_acc` | Rising from ~0.004 to ~1.0 | For each sample in the batch, was the correct future state the *highest-scoring* one?  Starts near 1/batch_size (random), converges toward 1.0. |
| `actor_loss` | Varies; no simple target | Sum of −Q and entropy penalty.  Not a simple indicator of progress. |
| `alpha` | 0.05 to 1.0 | Entropy temperature.  Adapts automatically.  If stuck high the policy is very random; if near 0 it is nearly deterministic. |
| `mean_reward` | 0 → 1+ | Average episode reward in recent training episodes.  Since reward is sparse (+1 only on success) this is essentially the success rate. |
| `eval/success_rate` | 0 → 1.0 | **The main signal**: fraction of evaluation episodes where the car reached the goal. |
| `eval/mean_ep_length` | 999 → shorter | Episodes end early on success, so shorter is better. |
| `steps/s` | 100–500 | Training throughput on CPU. |

### Typical learning curve

```
Steps       critic_acc    success_rate    What's happening
──────      ──────────    ────────────    ───────────────────────────────────
0–5 000     ~0.004        0 %             Random warmup, no updates yet
5–50 000    0.1–0.5       0 %             Embeddings learning environment geometry
50–150 000  0.5–0.9       0–30 %          Policy starting to exploit embeddings
150–300 000 0.8–1.0       30–80 %         Refinement toward consistent success
```

---

## 11. How This Relates to the Parent Repository

The parent SGCRL repository implements the same algorithm on complex robot
environments (Sawyer arm manipulation, maze navigation) with a heavily
engineered JAX/ACME infrastructure.  This folder strips all of that away:

| Parent (SGCRL) | This implementation |
|----------------|---------------------|
| JAX / Haiku networks | PyTorch `nn.Module` |
| ACME distributed actors / learners | Single-process loop in one file |
| Reverb replay buffer | ~40-line NumPy circular buffer |
| Sawyer robot / PointSpiral (needs MuJoCo) | `MountainCarContinuous-v0` (built into Gymnasium) |
| Contrastive NCE + CPC loss variants | Symmetric NCE (simplest variant) |
| Adaptive entropy (SAC) | Same, unchanged |
| Polyak target critic | Same, unchanged |

The **core learning algorithm is identical**.  Every concept you learn here
applies directly to the full SGCRL codebase — only the plumbing is different.
