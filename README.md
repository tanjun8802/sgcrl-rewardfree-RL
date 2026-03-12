# Single-goal Contrastive RL (SGCRL)

This repository contains the code for the paper **"A Single Goal is All You Need"**, which proposes SGCRL — a reward-free, goal-conditioned reinforcement learning method that trains an agent using only a **single fixed goal** per environment rather than sampling diverse goals during training. The core insight is that contrastive representation learning already encodes sufficient structure to generalise from one goal to the entire goal space.

---

## Table of Contents
1. [Set up conda environment](#set-up-conda-environment)
2. [Running Experiments](#running-experiments)
3. [Useful Flags Reference](#useful-flags-reference)
4. [Repository Pipeline](#repository-pipeline)
5. [File Descriptions Ranked by Importance](#file-descriptions-ranked-by-importance)

---

## Set up conda environment
**Set up conda:**
1. Load up anaconda: `module load anaconda3`
2. Clone the repository
3. Create an Anaconda environment: `conda create -n contrastive_rl python=3.9 -y`
4. Activate the environment: `conda activate contrastive_rl`

**Install package dependencies:**

1. Change library path: `export LD_LIBRARY_PATH={path to conda}/.conda/envs/contrastive_rl/lib/`
2. Install the requirements: `pip install -r requirements.txt --no-deps`
3. Download the mujoco binaries and place them in ~/.mujoco/ according to instructions in https://github.com/openai/mujoco-py. Run `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path to mujoco}/.mujoco/mujoco210/bin`
4. Reinstall strict versions for the following packages:
```
pip install dm-acme[jax,tf] 
pip install jax==0.4.10 jaxlib==0.4.10
pip install ml_dtypes==0.2.0
pip install dm-haiku==0.0.9
pip install gymnasium-robotics 
pip uninstall scipy; pip install scipy==1.12
pip install torch==2.1.2 scikit-learn pandas
```

**Potential errors and fixes:**
Cythonizing Error:

`fatal error: GL/glew.h: No such file or directory 4 | #include <GL/glew.h>`

Fix:
```
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
pip install patchelf
```

Cythinizing Error:

`Cannot assign type 'void (const char *) except * nogil' to 'void (*)(const char *) noexcept nogil'`

Fix: `pip install "cython<3"`

**Running with GPU:**

To enable GPU running, run these three commands in a shell with gpu access. This essentially picks out a set of gpu backend infrastructures that is simultaneously supported by jax and the repository code. Note that this step may vary depending on the specifics of the computing environment.

```
module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0
pip install optax==0.1.7
pip install --upgrade jax==0.4.7 jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/{path to cuda}/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

---

## Running Experiments

The single entry point for all experiments is `lp_contrastive.py`. Below are representative commands for all supported combinations.

### Quickstart (default settings)
```bash
# SGCRL with contrastive_cpc on sawyer_bin (fixed single goal — the paper's main setting)
python lp_contrastive.py
```

### Reproduce SGCRL paper experiments
```bash
# Replace <ENV> with: sawyer_bin | sawyer_box | sawyer_peg | point_Spiral11x11
python lp_contrastive.py --env='<ENV>' --alg='contrastive_cpc' --num_steps=8_000_000
```

Concrete examples:
```bash
python lp_contrastive.py --env='sawyer_bin'        --alg='contrastive_cpc' --num_steps=8_000_000
python lp_contrastive.py --env='sawyer_box'        --alg='contrastive_cpc' --num_steps=8_000_000
python lp_contrastive.py --env='sawyer_peg'        --alg='contrastive_cpc' --num_steps=8_000_000
python lp_contrastive.py --env='point_Spiral11x11' --alg='contrastive_cpc' --num_steps=8_000_000
```

### Run baseline / comparison algorithms
```bash
# NCE loss (contrastive_nce)
python lp_contrastive.py --alg='contrastive_nce'

# C-Learning (TD-based)
python lp_contrastive.py --alg='c_learning'

# NCE + C-Learning combined
python lp_contrastive.py --alg='nce+c_learning'
```

### Original Contrastive RL baseline (uniform goal sampling)
```bash
# This replicates the Eysenbach et al. 2022 setting by sampling goals uniformly
python lp_contrastive.py --sample_goals
```

### Logging and reproducibility
```bash
# Set random seed
python lp_contrastive.py --seed=0

# Save logs/checkpoints under a unique subfolder (useful for parallel sweeps)
python lp_contrastive.py --add_uid

# Change log directory
python lp_contrastive.py --log_dir_path='my_experiments/'

# Change checkpoint frequency (in minutes)
python lp_contrastive.py --time_delta_minutes=10
```

### Multi-threaded execution
```bash
# Run using multi-threading (launchpad local_mt backend)
python lp_contrastive.py --lp_launch_type=local_mt
```

### Reading logs
Training logs (CSV format) are saved to `logs/<alg>_<env>_<seed>/` by default (or the path given by `--log_dir_path`). Each run creates:
- `learner.csv` — critic loss, actor loss, accuracy metrics, steps/second
- `actor.csv` — episode success rate, distance to goal
- `evaluator.csv` — evaluation success rate and distance metrics on the fixed goal

You can monitor training with TensorBoard if you point it at the log directory:
```bash
tensorboard --logdir logs/
```

---

## Useful Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `sawyer_bin` | Environment name. Options: `sawyer_bin`, `sawyer_box`, `sawyer_peg`, `point_Spiral11x11` |
| `--alg` | `contrastive_cpc` | Algorithm. Options: `contrastive_nce`, `contrastive_cpc`, `c_learning`, `nce+c_learning` |
| `--num_steps` | `8_000_000` | Total number of actor environment steps (8 million) |
| `--sample_goals` | `False` | If set, goals are sampled uniformly (original Contrastive RL baseline) |
| `--seed` | `42` | Random seed for reproducibility |
| `--add_uid` | `False` | Append a unique ID to the log/checkpoint directory name |
| `--log_dir_path` | `logs/` | Root directory for logs and checkpoints |
| `--time_delta_minutes` | `5` | How often (in minutes) to save checkpoints |

---

## Repository Pipeline

The training system uses **DeepMind Launchpad** to orchestrate multiple concurrent processes. The high-level pipeline is:

```
┌─────────────────────────────────────────────────────────────────────┐
│  lp_contrastive.py  (entry point)                                   │
│    • Parses flags, builds params dict                               │
│    • Constructs DistributedContrastive agent via get_program()      │
│    • Launches all processes with lp.launch()                        │
└───────────┬─────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│  Distributed Layout (contrastive/distributed_layout.py)                       │
│                                                                               │
│   ┌─────────────┐    episodes     ┌──────────────────────────────────────┐   │
│   │  N Actors   │ ─────────────►  │  Reverb Replay Buffer (dm-reverb)    │   │
│   │ (env loops) │                 │  Stores full episodes as trajectories │   │
│   └──────┬──────┘                 └──────────────────┬───────────────────┘   │
│          │ policy params                             │ sampled batches        │
│          │ (VariableClient)                          │                        │
│          ▼                                           ▼                        │
│   ┌─────────────┐                 ┌──────────────────────────────────────┐   │
│   │  Evaluator  │ ◄── params ──   │  Learner (ContrastiveLearner)        │   │
│   │ (fixed goal │                 │  • flatten_fn: sample (s,a,g,s') from│   │
│   │  eval loop) │                 │    episodes using future-state goals  │   │
│   └─────────────┘                 │  • critic_loss (NCE/CPC/TD)          │   │
│                                   │  • actor_loss (maximize Q-diagonal)  │   │
│   ┌─────────────┐                 │  • alpha_loss (entropy temperature)  │   │
│   │  Counter /  │                 │  • soft target-network update (τ)    │   │
│   │  Checkpoint │                 └──────────────────────────────────────┘   │
│   └─────────────┘                                                             │
└───────────────────────────────────────────────────────────────────────────────┘
```

### Data flow in detail

1. **Environment step** — An actor queries the policy, receives `(obs, action, reward, next_obs)`, and pushes full episodes into Reverb.
2. **Replay sampling** — `make_dataset_iterator` in `builder.py` reads trajectories and applies `flatten_fn`: for each transition `(s_t, a_t)` it samples a *future* state `s_f` from the same episode and forms the goal-augmented observation `[s_t ‖ goal(s_f)]`.
3. **Critic update** — The Q-network computes a `(batch × batch)` matrix of inner products between `(s, a)` and goal representations. The diagonal entries correspond to matched pairs. The loss is NCE (cross-entropy over rows) or CPC (with a logsumexp regulariser) or a TD/Bellman variant.
4. **Actor update** — The policy is updated to maximise `diag(Q(s, π(s|g), g))` minus an entropy bonus.
5. **Target network** — A soft copy of the critic is maintained and used for bootstrap targets in the TD variant.
6. **Evaluation** — A separate evaluator process runs episodes with the *fixed* goal and logs success rate and L2 distance to goal.

### Goal-conditioning strategy

| Flag | Goal during training | Goal during evaluation |
|------|----------------------|------------------------|
| `--sample_goals` off (default, SGCRL) | Single fixed goal per environment | Fixed goal |
| `--sample_goals` on (baseline) | Uniform random goal per episode | Fixed goal |

---

## File Descriptions Ranked by Importance

### Rank 1 — `lp_contrastive.py`
⭐⭐⭐⭐⭐ **Entry point.** Parses all command-line flags, builds the parameter dictionary, instantiates the `DistributedContrastive` agent, and launches all distributed processes via `lp.launch()`. This is the only file you need to call directly. It also defines the `fixed_goal_dict` — the single fixed goal coordinate for each supported environment, which is the defining feature of SGCRL.

### Rank 2 — `contrastive/learning.py`
⭐⭐⭐⭐⭐ **Core learning algorithm.** Implements `ContrastiveLearner`, which contains:
- `critic_loss`: the contrastive Q-function loss (NCE, CPC, or TD/C-Learning variants).
- `actor_loss`: SAC-style policy gradient using the diagonal of the Q-matrix.
- `alpha_loss`: adaptive entropy temperature (SAC temperature tuning).
- `update_step`: one full gradient step (critic → target update → actor → optional alpha).
- `TrainingState`: a `NamedTuple` holding all learnable parameters and optimiser states.

### Rank 3 — `contrastive/networks.py`
⭐⭐⭐⭐ **Network architecture.** Defines `ContrastiveNetworks` and `make_networks`:
- `sa_encoder` MLP: encodes `(state, action)` → `repr_dim`-dimensional vector.
- `g_encoder` MLP: encodes `goal` → `repr_dim`-dimensional vector.
- `_combine_repr`: computes the outer-product (dot-product matrix) between SA-repr and goal-repr batches. The diagonal of this matrix is used as the Q-values.
- `_actor_fn`: MLP policy head outputting a `NormalTanhDistribution`.
- Optional `twin_q`: runs two independent critic heads and takes the minimum.

### Rank 4 — `contrastive/builder.py`
⭐⭐⭐⭐ **Glue between replay and learner.** Implements `ContrastiveBuilder`:
- `make_replay_tables`: configures a Reverb table with a `SampleToInsertRatio` rate limiter.
- `make_dataset_iterator`: reads episodes from Reverb and applies `flatten_fn` to produce goal-conditioned `(s, a, g, s')` transitions. Future goals are sampled proportionally to a discounted distribution over the episode horizon.
- `make_learner`: instantiates `ContrastiveLearner` with Adam optimisers.
- `make_actor`: wraps the policy in a `GenericActor` (or `InitiallyRandomActor` for the warm-start phase).
- `make_adder`: configures an `EpisodeAdder` that stores full episodes.

### Rank 5 — `contrastive/distributed_layout.py`
⭐⭐⭐ **Distributed orchestration.** Implements `DistributedLayout`, which defines each Launchpad node as a method:
- `replay()`: Reverb node.
- `learner()`: learner node (wraps `ContrastiveLearner` in a `CheckpointingRunner`).
- `actor()`: actor node (environment loop + adder).
- `evaluator()`: evaluation loop with fixed goals.
- `counter()`: global step counter with checkpointing.
- `coordinator()`: `StepsLimiter` that stops training when `max_number_of_steps` is reached.
- `build()`: wires all nodes into a Launchpad `Program`.

### Rank 6 — `contrastive/agents.py`
⭐⭐⭐ **Agent class.** Implements `DistributedContrastive`, a thin subclass of `DistributedLayout` that:
- Assembles the evaluator with `SuccessObserver` and `DistanceObserver`.
- Constructs logging paths based on `alg_name`, `env_name`, and `seed`.
- Passes the fixed-goal environment factory to the evaluator.

### Rank 7 — `env_utils.py`
⭐⭐⭐ **Environment loading and wrappers.** Contains:
- `load()`: factory that instantiates the correct environment class given `env_name`.
- `SawyerBin`, `SawyerBox`, `SawyerPeg`: MetaWorld-based robotic manipulation environments. Each overrides `reset()`, `step()`, and `_get_obs()` to support fixed or randomly sampled goals and return `[obs ‖ goal]`-concatenated observations.
- The SGCRL vs. baseline distinction is realised here: when `fixed_start_end` is not `None`, the goal is always the paper's fixed coordinate; otherwise a random goal is sampled each episode.

### Rank 8 — `point_env.py`
⭐⭐ **2-D navigation environment.** Implements `PointEnv`, a simple `gym.Env` for maze navigation. Multiple wall layouts are defined (Small, Cross, FourRooms, Spiral11x11, …). The agent moves a point mass through the maze, and the reward is 1 if within distance 1.0 of the goal. Used for fast debugging and ablation studies.

### Rank 9 — `contrastive/config.py`
⭐⭐ **Hyperparameter dataclass.** `ContrastiveConfig` is a `@dataclasses.dataclass` that centralises all hyperparameters:
- Replay sizes, batch size, discount, learning rates, target smoothing coefficient (τ).
- Algorithm switches: `use_cpc`, `use_td`, `twin_q`, `add_mc_to_td`.
- Environment metadata set at runtime: `obs_dim`, `max_episode_steps`, `start_index`, `end_index`.
- `target_entropy_from_env_spec()`: helper to compute a default entropy target for SAC.

### Rank 10 — `distributional.py`
⭐⭐ **Policy distribution head.** Defines `NormalTanhDistribution` (used in `networks.py` as the actor output head) and several supporting Haiku modules (`TanhTransformedDistribution`, `GaussianMixture`, `CategoricalHead`, `MultivariateNormalDiagHead`, `DiscreteValued`). The key module is `NormalTanhDistribution`, which outputs a `Normal` distribution passed through a `Tanh` bijector to keep actions in `[-1, 1]`.

### Rank 11 — `default.py`
⭐ **Logging factory.** `make_default_logger` constructs a composite logger that writes to the terminal and, optionally, to a CSV file. Used by the learner, actor, and evaluator nodes for metric logging.

### Rank 12 — `contrastive/utils.py`
⭐ **Utility helpers.** Contains:
- `obs_to_goal_1d` / `obs_to_goal_2d`: slices the goal coordinates out of an observation vector.
- `make_environment`: wraps a raw gym environment with `GymWrapper`, `StepLimitWrapper`, and `ObservationFilterWrapper` to produce a `dm_env.Environment` compatible with Acme.
- `SuccessObserver`: tracks whether any reward was positive in an episode (binary success).
- `DistanceObserver`: tracks L2 distance to goal across an episode (init, final, delta, min).
- `InitiallyRandomActor`: acts uniformly at random until the first learner update arrives (warm-start).
- `ObservationFilterWrapper`: strips unused observation coordinates.
