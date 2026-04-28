"""
Training script for Contrastive Goal-Conditioned RL on MountainCarContinuous-v0.

Quick start
-----------
    # Install dependencies (only PyTorch + Gymnasium needed)
    pip install torch gymnasium

    # Run with default settings
    python -m mountain_car_gcrl.train

    # Evaluate a saved checkpoint
    python -m mountain_car_gcrl.train --eval_only --checkpoint runs/best_actor.pt

Usage
-----
    python -m mountain_car_gcrl.train [OPTIONS]

Options (all have sensible defaults):
    --total_steps INT       Total environment steps          [default: 300_000]
    --batch_size INT        Mini-batch size                  [default: 256]
    --replay_capacity INT   Replay buffer capacity           [default: 500_000]
    --warmup_steps INT      Random exploration before training [default: 5_000]
    --updates_per_step INT  Gradient updates per env step    [default: 1]
    --eval_freq INT         Evaluate every N steps           [default: 5_000]
    --eval_episodes INT     Episodes per evaluation          [default: 10]
    --save_dir STR          Directory to save checkpoints    [default: "runs/"]
    --seed INT              Random seed                      [default: 42]
    --device STR            "cpu" or "cuda"                  [default: "cpu"]
    --eval_only             Skip training; run evaluation only
    --checkpoint STR        Path to a checkpoint to load
    --render                Render during evaluation
"""

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from mountain_car_gcrl.agent       import ContrastiveGCRL
from mountain_car_gcrl.env_wrapper import GoalConditionedMountainCar
from mountain_car_gcrl.replay_buffer import ReplayBuffer


# ---------------------------------------------------------------------- #
# Logging helpers
# ---------------------------------------------------------------------- #

class Logger:
    """Lightweight console + CSV logger."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self._csv_path  = os.path.join(log_dir, "training_log.csv")
        self._headers_written = False

    def log(self, step: int, metrics: dict, prefix: str = "") -> None:
        """Print metrics and append a row to the CSV file."""
        label = f"[step {step:>7d}]"
        if prefix:
            label += f" [{prefix}]"
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in metrics.items()]
        print(label, "  ".join(parts))

        row = {"step": step, **metrics}
        if not self._headers_written:
            with open(self._csv_path, "w") as f:
                f.write(",".join(row.keys()) + "\n")
            self._headers_written = True
        with open(self._csv_path, "a") as f:
            f.write(",".join(str(v) for v in row.values()) + "\n")


# ---------------------------------------------------------------------- #
# Evaluation rollout
# ---------------------------------------------------------------------- #

def evaluate(agent: ContrastiveGCRL, n_episodes: int = 10, render: bool = False) -> dict:
    """Run ``n_episodes`` deterministic rollouts and return summary metrics.

    The goal is fixed to the default top-of-hill target during evaluation so
    results are comparable across checkpoints.
    """
    render_mode = "human" if render else None
    env = GoalConditionedMountainCar(render_mode=render_mode)

    successes      = 0
    total_rewards  = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_reward = 0.0
        ep_steps  = 0

        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done       = terminated or truncated
            ep_reward += reward
            ep_steps  += 1

        total_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        if info.get("reached_goal", False):
            successes += 1

    env.close()

    return {
        "eval/success_rate":   successes / n_episodes,
        "eval/mean_reward":    float(np.mean(total_rewards)),
        "eval/mean_ep_length": float(np.mean(episode_lengths)),
    }


# ---------------------------------------------------------------------- #
# Main training loop
# ---------------------------------------------------------------------- #

def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------ #
    # Reproducibility
    # ------------------------------------------------------------------ #
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir)

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    # Training env: sample random goals to encourage broad exploration
    train_env = GoalConditionedMountainCar(sample_random_goals=True)
    obs, _    = train_env.reset(seed=args.seed)

    state_dim  = train_env.state_dim   # 2
    goal_dim   = train_env.goal_dim    # 2
    action_dim = train_env.action_space.shape[0]  # 1
    obs_dim    = state_dim + goal_dim             # 4

    print(f"\nEnvironment: MountainCarContinuous-v0  (goal-conditioned)")
    print(f"  state_dim={state_dim}  goal_dim={goal_dim}  action_dim={action_dim}")
    print(f"  obs_dim (state+goal) = {obs_dim}\n")

    # ------------------------------------------------------------------ #
    # Agent & replay buffer
    # ------------------------------------------------------------------ #
    agent = ContrastiveGCRL(
        state_dim    = state_dim,
        action_dim   = action_dim,
        goal_dim     = goal_dim,
        hidden_sizes = (256, 256),
        repr_dim     = 64,
        device       = args.device,
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    if args.eval_only:
        print("\n--- Evaluation only mode ---")
        metrics = evaluate(agent, args.eval_episodes, args.render)
        logger.log(0, metrics, prefix="eval")
        return

    replay = ReplayBuffer(
        obs_dim    = obs_dim,
        action_dim = action_dim,
        capacity   = args.replay_capacity,
    )

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    print(f"Starting training for {args.total_steps:,} steps ...\n")
    print(f"  Warmup period: {args.warmup_steps:,} random steps before first update")
    print(f"  Evaluation every {args.eval_freq:,} steps\n")

    best_success_rate = 0.0
    recent_ep_rewards = deque(maxlen=20)
    ep_reward  = 0.0
    ep_steps   = 0
    ep_num     = 0
    train_metrics_buffer = []  # accumulate metrics between log events

    start_time = time.time()

    for total_step in range(1, args.total_steps + 1):
        # ---- Collect one transition ------------------------------------ #
        if total_step <= args.warmup_steps:
            # Random exploration at the start to fill the replay buffer
            action = train_env.action_space.sample()
        else:
            action = agent.select_action(obs)

        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated

        replay.add(obs, action, reward, next_obs, float(done))

        obs        = next_obs
        ep_reward += reward
        ep_steps  += 1

        if done:
            obs, _ = train_env.reset()
            recent_ep_rewards.append(ep_reward)
            ep_reward = 0.0
            ep_steps  = 0
            ep_num   += 1

        # ---- Gradient update ------------------------------------------ #
        if total_step > args.warmup_steps and replay.ready:
            for _ in range(args.updates_per_step):
                batch   = replay.sample(args.batch_size)
                metrics = agent.update(batch)
                train_metrics_buffer.append(metrics)

        # ---- Periodic console log ------------------------------------- #
        if total_step % 5_000 == 0:
            avg = lambda key: float(np.mean([m[key] for m in train_metrics_buffer])) \
                              if train_metrics_buffer else 0.0
            elapsed = time.time() - start_time
            log_metrics = {
                "episodes":     ep_num,
                "critic_loss":  avg("critic_loss"),
                "actor_loss":   avg("actor_loss"),
                "alpha":        avg("alpha"),
                "critic_acc":   avg("critic_acc"),
                "mean_reward":  float(np.mean(recent_ep_rewards)) if recent_ep_rewards else 0.0,
                "steps/s":      total_step / elapsed,
            }
            logger.log(total_step, log_metrics, prefix="train")
            train_metrics_buffer.clear()

        # ---- Periodic evaluation -------------------------------------- #
        if total_step % args.eval_freq == 0:
            eval_metrics = evaluate(agent, args.eval_episodes)
            logger.log(total_step, eval_metrics, prefix="eval")

            success_rate = eval_metrics["eval/success_rate"]
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                ckpt_path = os.path.join(args.save_dir, "best_actor.pt")
                agent.save(ckpt_path)
                print(f"  ✓ New best success rate {success_rate:.2%} → saved {ckpt_path}")

    # Final checkpoint
    agent.save(os.path.join(args.save_dir, "final_actor.pt"))
    print(f"\nTraining complete.  Best success rate: {best_success_rate:.2%}")
    train_env.close()


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Contrastive GCRL on MountainCarContinuous-v0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--total_steps",    type=int,   default=300_000)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--replay_capacity",type=int,   default=500_000)
    p.add_argument("--warmup_steps",   type=int,   default=5_000)
    p.add_argument("--updates_per_step",type=int,  default=1)
    p.add_argument("--eval_freq",      type=int,   default=5_000)
    p.add_argument("--eval_episodes",  type=int,   default=10)
    p.add_argument("--save_dir",       type=str,   default="runs/")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         type=str,   default="cpu",
                   choices=["cpu", "cuda"])
    p.add_argument("--eval_only",      action="store_true",
                   help="Skip training and only run evaluation")
    p.add_argument("--checkpoint",     type=str,   default=None,
                   help="Path to a checkpoint to load before training/eval")
    p.add_argument("--render",         action="store_true",
                   help="Render the environment during evaluation")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
