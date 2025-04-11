from abc import ABC, abstractmethod

import numpy as np


class MyAgent(ABC):
    @abstractmethod
    def test(self, env, argmax=True, reset_graph=True):
        pass


class RandomAgent(MyAgent):
    def test(self, env, argmax=True, reset_graph=True):
        terminal = False
        observation, mask = env.reset(reset_graph)
        rewards = []
        actions_res = []
        masks_res = []
        while not terminal:
            action = np.random.choice(np.arange(mask.shape[0])[mask])

            masks_res.append(mask)
            observation, mask, reward, terminal, _ = env.step(action)
            rewards.append(reward)
            actions_res.append(action)

        return env.compute_objective_function(), env.graph.edge_attr[:, 1], rewards, actions_res  # , masks_res

class OptimalAgent(MyAgent):
    def test(self, env, argmax=True, reset_graph=True):
        return env.calculate_min_span_tree(), [], [1], []  # , masks_res