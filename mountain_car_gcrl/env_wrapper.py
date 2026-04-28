"""
Goal-conditioned wrapper for MountainCarContinuous-v0.

The standard MountainCarContinuous-v0 environment has:
  - State:  [position, velocity]  (shape: 2,)
  - Action: [force]               (shape: 1,  range [-1, 1])

We wrap it so that the agent receives a goal-augmented observation:
  - Observation: [position, velocity, goal_position, goal_velocity]  (shape: 4,)

The goal is the target state the agent should reach.  By default the goal is
[0.45, 0.0], i.e. the flag at the top of the hill with zero velocity, which
is also the condition that triggers the original +100 reward.

You can sample random goals (useful during training) or use a fixed goal
(useful for evaluation).
"""

import gymnasium as gym
import numpy as np


# The position of the flag / success threshold in MountainCarContinuous-v0
SUCCESS_POSITION = 0.45
SUCCESS_VELOCITY = 0.0
DEFAULT_GOAL = np.array([SUCCESS_POSITION, SUCCESS_VELOCITY], dtype=np.float32)


class GoalConditionedMountainCar(gym.Wrapper):
    """Wraps MountainCarContinuous-v0 with a goal-conditioned observation space.

    The observation fed to the agent is the *concatenation* of the current
    state and the desired goal:
        obs = [position, velocity, goal_position, goal_velocity]

    This mirrors the design of the original SGCRL code where the observation
    is always [state | goal].
    """

    def __init__(
        self,
        goal: np.ndarray | None = None,
        sample_random_goals: bool = False,
        max_episode_steps: int = 999,
        render_mode: str | None = None,
    ):
        """
        Args:
            goal: Fixed goal array [goal_pos, goal_vel].  If None and
                  ``sample_random_goals`` is False, the default top-of-hill
                  goal is used.
            sample_random_goals: If True, a new goal is sampled uniformly
                  from the position range [-0.6, 0.6] at each episode reset.
                  Useful for training; disabled during evaluation.
            max_episode_steps: Episode length cap.
            render_mode: Passed to the underlying gym environment (e.g.
                  "human" or "rgb_array").
        """
        env = gym.make(
            "MountainCarContinuous-v0",
            max_episode_steps=max_episode_steps,
            render_mode=render_mode,
        )
        super().__init__(env)

        self.sample_random_goals = sample_random_goals
        self._fixed_goal = DEFAULT_GOAL if goal is None else np.array(goal, dtype=np.float32)
        self._current_goal = self._fixed_goal.copy()

        # State dimension of the *unwrapped* environment (position + velocity)
        self.state_dim = env.observation_space.shape[0]   # 2
        self.goal_dim  = self._fixed_goal.shape[0]         # 2

        # New observation space: [state | goal]
        low  = np.concatenate([env.observation_space.low,  -np.ones(self.goal_dim)])
        high = np.concatenate([env.observation_space.high,  np.ones(self.goal_dim)])
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)

        if self.sample_random_goals:
            # Sample a reachable goal position (right half of the range)
            goal_pos = self.np_random.uniform(0.1, 0.6)
            goal_vel = 0.0
            self._current_goal = np.array([goal_pos, goal_vel], dtype=np.float32)
        else:
            self._current_goal = self._fixed_goal.copy()

        obs = self._make_obs(state)
        return obs, info

    def step(self, action):
        state, _, terminated, truncated, info = self.env.step(action)

        obs = self._make_obs(state)

        # Reward: we ignore the environment's shaped reward and instead give
        # a sparse +1 only when the agent reaches the goal (goal-conditioned).
        reached = self._reached_goal(state)
        reward  = 1.0 if reached else 0.0

        # End the episode early on success so the agent gets a clean signal.
        terminated = terminated or reached

        info["reached_goal"] = reached
        info["goal"]         = self._current_goal.copy()
        info["state"]        = state.copy()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs(self, state: np.ndarray) -> np.ndarray:
        """Concatenate [state | goal] into a single observation vector."""
        return np.concatenate([state, self._current_goal], dtype=np.float32)

    def _reached_goal(self, state: np.ndarray) -> bool:
        """Return True when the car is close enough to the goal."""
        pos_threshold = 0.05   # within 5 cm of goal position
        vel_threshold = 0.05   # within 0.05 m/s of goal velocity
        return (
            abs(state[0] - self._current_goal[0]) < pos_threshold
            and abs(state[1] - self._current_goal[1]) < vel_threshold
        )

    @property
    def current_goal(self) -> np.ndarray:
        return self._current_goal.copy()
