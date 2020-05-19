from typing import Optional

import gym
import numpy as np


class CorridorEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)

    def __init__(self, length: int = 10):
        super().__init__()
        self._length = length
        self._agent: Optional[int] = None
        self._goal: Optional[int] = None

        self.observation_space = gym.spaces.MultiBinary(self._length)

    def reset(self) -> np.ndarray:
        self._agent, self._goal = np.random.choice(self._length, 2)
        return self._prepare_obs()

    def step(self, action: int) -> (np.ndarray, float, bool, dict):
        if action == 0 and self._agent > 0:
            self._agent -= 1
        elif action == 1 and self._agent < (self._length - 1):
            self._agent += 1
        done = self._agent == self._goal
        reward = 1.0 if done else 0.0
        return self._prepare_obs(), reward, done, {}

    def render(self, mode: str = 'human'):
        print(self._prepare_obs())

    def _prepare_obs(self) -> np.ndarray:
        corridor = np.zeros(self._length * 2 - 1, dtype=np.float32)
        distance_to_goal = self._goal - self._agent
        corridor[self._length - 1 + distance_to_goal] = 1
        return corridor
